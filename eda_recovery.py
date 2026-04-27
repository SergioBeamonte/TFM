"""
IDRecovery
===================================
"""

import os
import csv
import warnings
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pysmile
import pysmile_license

from EDAspy.optimization.univariate import UMDAc

from extractor import NetworkExtractor
from engines import ShachterEngine
from models import NodeKind


# ─────────────────────────────────────────────────────────────────────────────
# Utilidades auxiliares
# ─────────────────────────────────────────────────────────────────────────────

def topological_sort(nodes: Dict) -> List[str]:
    """
    Devuelve los nombres de los nodos en orden topológico (padres antes que hijos).
    Uasa el algoritmo de Kahn sobre el grafo de dependencias implícito en .parents.
    """
    in_degree = {name: 0 for name in nodes}
    children = defaultdict(list)

    for name, nd in nodes.items():
        for parent in getattr(nd, 'parents', []):
            if parent in nodes:
                in_degree[name] += 1
                children[parent].append(name)

    queue = [n for n, d in in_degree.items() if d == 0]
    order = []
    while queue:
        node = queue.pop(0)
        order.append(node)
        for child in children[node]:
            in_degree[child] -= 1
            if in_degree[child] == 0:
                queue.append(child)

    if len(order) != len(nodes):
        warnings.warn("Ciclo detectado en el grafo — ordenación topológica incompleta.")
    return order


def softmax_rows(arr: np.ndarray) -> np.ndarray:
    """
    Aplica softmax a cada fila del array (última dimensión).
    Más estable numéricamente que la división simple y evita división por cero.
    """
    shifted = arr - arr.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def apply_sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Transforma reales en (-inf, +inf) al rango (0, 1).
    Usado exclusivamente para acotar las variables independientes de utilidad.
    """
    return 1.0 / (1.0 + np.exp(-x))


# ─────────────────────────────────────────────────────────────────────────────
# Clase principal
# ─────────────────────────────────────────────────────────────────────────────

class IDRecovery:
    """
    Reconstruye los parámetros de un Diagrama de Influencia de estructura fija
    a partir de un conjunto de reglas expertas, usando EDAs.
    """

    def __init__(
        self,
        xdsl_path: str,
        rules_csv: str,
        best_util_config: Dict[str, str],
        worst_util_config: Dict[str, str],
        rule_weights: Optional[List[float]] = None,
        util_range: Tuple[float, float] = (0.0, 10.0),
    ):
        self.xdsl_path = xdsl_path
        self.rules_csv = rules_csv
        self.best_util_config = best_util_config
        self.worst_util_config = worst_util_config
        self.util_min, self.util_max = util_range

        self.net = pysmile.Network()
        self.net.read_file(xdsl_path)
        self.nodes = NetworkExtractor.extract(self.net)

        self.rules = self._parse_rules(rules_csv)
        if rule_weights is not None:
            self.rule_weights = np.array(rule_weights, dtype=float)
        else:
            self.rule_weights = np.ones(len(self.rules), dtype=float)
        self.rule_weights /= self.rule_weights.sum()

        self.topo_order = topological_sort(self.nodes)
        self.param_specs = self._get_param_specs()
        self.total_vars = sum(spec['free_size'] for spec in self.param_specs)
        self.engine = ShachterEngine()

        print(f"[+] Reglas cargadas:           {len(self.rules)}")
        print(f"[+] Nodos en orden topológico: {self.topo_order}")
        print(f"[+] Variables libres a opti.:  {self.total_vars}")
        print("[+] Modo de optimización: Continuo (Softmax para CPTs, Sigmoide para Utilidad)")

    # ──────────────────────────────────────────────────────────────────────────
    # Parseo de reglas
    # ──────────────────────────────────────────────────────────────────────────

    def _parse_rules(self, csv_path: str) -> List[Tuple]:
        rules = []
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row_num, row in enumerate(reader, start=2):
                evidence: Dict[str, str] = {}
                target_decision: Optional[str] = None
                target_action: Optional[str] = None

                for node_name, val_str in row.items():
                    val_str = val_str.strip()
                    if not val_str:
                        continue
                    try:
                        val = int(val_str)
                    except ValueError:
                        warnings.warn(
                            f"Fila {row_num}, columna '{node_name}': "
                            f"valor '{val_str}' no es entero — ignorado."
                        )
                        continue

                    if val == 0:
                        continue

                    if node_name not in self.nodes:
                        warnings.warn(f"Columna '{node_name}' no encontrada en el modelo.")
                        continue

                    nd = self.nodes[node_name]
                    if val > 0:
                        state_idx = val - 1
                        if state_idx >= len(nd.states):
                            warnings.warn(
                                f"Fila {row_num}: estado {state_idx} fuera de rango "
                                f"para nodo '{node_name}' ({len(nd.states)} estados)."
                            )
                            continue
                        evidence[node_name] = nd.states[state_idx]
                    else:  # val < 0
                        state_idx = abs(val) - 1
                        if state_idx >= len(nd.states):
                            warnings.warn(
                                f"Fila {row_num}: decisión {state_idx} fuera de rango "
                                f"para nodo '{node_name}'."
                            )
                            continue
                        target_decision = node_name
                        target_action = nd.states[state_idx]

                if target_decision and evidence:
                    rules.append((evidence, target_decision, target_action))
                else:
                    warnings.warn(f"Fila {row_num}: regla incompleta (sin evidencia o sin decisión) — ignorada.")

        return rules

    # ──────────────────────────────────────────────────────────────────────────
    # Especificaciones de parámetros
    # ──────────────────────────────────────────────────────────────────────────

    def _get_fixed_util_indices(self, nd) -> List[tuple]:
        """Devuelve los índices numpy de las entradas fijas de la tabla de utilidad."""
        fixed = []
        for config in [self.best_util_config, self.worst_util_config]:
            try:
                idx = tuple(self.nodes[p].states.index(config[p]) for p in nd.parents)
                fixed.append(idx)
            except (KeyError, ValueError):
                pass
        return fixed

    def _get_param_specs(self) -> List[Dict]:
        """
        Construye las especificaciones de todos los bloques de parámetros libres,
        en orden topológico. Para la utilidad, excluye las entradas fijas.
        """
        specs = []
        for name in self.topo_order:
            nd = self.nodes[name]
            if nd.kind == NodeKind.CHANCE:
                specs.append({
                    'name': name,
                    'kind': 'chance',
                    'size': nd.table.size,
                    'free_size': nd.table.size,
                    'shape': nd.table.shape,
                    'fixed_values': [],
                })
            elif nd.kind == NodeKind.UTILITY:
                fixed_indices = self._get_fixed_util_indices(nd)
                fixed_values = []
                flat_fixed_positions = set()

                for idx in fixed_indices:
                    is_best = (idx == tuple(self.nodes[p].states.index(self.best_util_config[p]) for p in nd.parents) if self.best_util_config else False)
                    val = self.util_max if is_best else self.util_min
                    fixed_values.append((idx, val))
                    flat_fixed_positions.add(np.ravel_multi_index(idx, nd.table.shape))

                free_size = nd.table.size - len(flat_fixed_positions)
                free_mask = np.ones(nd.table.size, dtype=bool)
                for pos in flat_fixed_positions:
                    free_mask[pos] = False

                specs.append({
                    'name': name,
                    'kind': 'utility',
                    'size': nd.table.size,
                    'free_size': free_size,
                    'shape': nd.table.shape,
                    'fixed_values': fixed_values,
                    'free_mask': free_mask,
                })
        return specs

    # ──────────────────────────────────────────────────────────────────────────
    # Traducción vector → parámetros del modelo
    # ──────────────────────────────────────────────────────────────────────────

    def _vector_to_nodes(self, vector: np.ndarray):
        """
        Traduce el vector del optimizador en parámetros del modelo basándose en 
        la naturaleza del nodo (Softmax para Chance, Sigmoide para Utility).
        """
        pos = 0
        for spec in self.param_specs:
            free_size = spec['free_size']
            raw = vector[pos: pos + free_size].copy()
            pos += free_size
            name = spec['name']
            nd = self.nodes[name]

            if spec['kind'] == 'chance':
                reshaped = raw.reshape(spec['shape'])
                # Softmax hace que las clases sumen exactamente 1 por fila
                normalized = softmax_rows(reshaped)
                nd.table = normalized
                h = self.net.get_node(name)
                self.net.set_node_definition(h, normalized.flatten().tolist())

            elif spec['kind'] == 'utility':
                full_flat = np.zeros(spec['size'])
                # Aplicamos Sigmoide para mantenerlo entre 0 y 1, luego escalamos a (Min, Max)
                free_vals = apply_sigmoid(raw) * (self.util_max - self.util_min) + self.util_min
                full_flat[spec['free_mask']] = free_vals

                reshaped = full_flat.reshape(spec['shape'])
                for idx, val in spec['fixed_values']:
                    reshaped[idx] = val
                nd.table = reshaped
                h = self.net.get_node(name)
                self.net.set_node_definition(h, reshaped.flatten().tolist())

    # ──────────────────────────────────────────────────────────────────────────
    # Función de Aptitud
    # ──────────────────────────────────────────────────────────────────────────

    def fitness(self, vector: np.ndarray) -> float:
        """
        Función de aptitud. El vector entra en continuo real.
        """
        self._vector_to_nodes(vector)

        weighted_correct = 0.0
        for (evidence, target_node, expected_action), weight in zip(self.rules, self.rule_weights):
            try:
                res = self.engine.evaluate(self.nodes, evidence, self.net)
                if res.optimal_decisions.get(target_node) == expected_action:
                    weighted_correct += weight
            except Exception:
                # Falla la inferencia -> La regla no cuenta, se asume penalización
                pass

        return 1.0 - weighted_correct

    # ──────────────────────────────────────────────────────────────────────────
    # Ejecución del EDA
    # ──────────────────────────────────────────────────────────────────────────
    
    def run(
        self,
        size_gen: int = 100,
        max_iter: int = 50,
        dead_iter: int = 15,
        alpha: float = 0.5,
        verbose: bool = True,
        log_csv: str = "convergencia_umda.csv"
    ):
        """
        Ejecuta UMDAc para encontrar los parámetros que satisfacen las reglas.
        """
        # Limites fijos para que el espacio de búsqueda se adapte perfectamente 
        # tanto a la sigmoide como al softmax (-6 a 6 cubre ~99.8% de la varianza).
        lower, upper = -6.0, 6.0

        umda = UMDAc(
            size_gen=size_gen,
            max_iter=max_iter,
            dead_iter=dead_iter,
            n_variables=self.total_vars,
            alpha=alpha,
            lower_bound=lower,
            upper_bound=upper,
        )

        if verbose:
            print(f"\n[*] Iniciando UMDAc:")
            print(f"    Población: {size_gen} | Iteraciones: {max_iter} | Dead: {dead_iter}")
            print(f"    Variables libres: {self.total_vars}")
            print(f"    Reglas: {len(self.rules)}")

        # Ejecutamos el motor
        result = umda.minimize(self.fitness, verbose=verbose)

        accuracy = 1.0 - result.best_cost
        print(f"\n{'─'*50}")
        print(f"  Optimización finalizada")
        print(f"  Precisión final: {accuracy:.1%} ({int(accuracy * len(self.rules))}/{len(self.rules)} reglas)")
        print(f"  Generaciones ejecutadas: {result.n_iter}")
        print(f"{'─'*50}")

        # Aplicar la mejor solución y diagnosticar
        self._vector_to_nodes(result.best_ind)
        self._print_diagnostics(result)

        # Exportación a CSV de la convergencia
        if log_csv:
            with open(log_csv, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['generacion', 'mejor_coste', 'precision'])
                for i, cost in enumerate(result.history):
                    writer.writerow([i + 1, cost, 1.0 - cost])
            print(f"[+] Historial de convergencia guardado en: {log_csv}")

        return result

    # ──────────────────────────────────────────────────────────────────────────
    # Diagnóstico
    # ──────────────────────────────────────────────────────────────────────────
    
    def _print_diagnostics(self, result):
        """
        Muestra los valores finales óptimos que el algoritmo ha encontrado
        para cada nodo tras aplicar las transformaciones correctas (Softmax/Sigmoide).
        """
        print("\n[DIAGNÓSTICO] Valores óptimos encontrados por las reglas:\n")

        pos = 0
        for spec in self.param_specs:
            free_size = spec['free_size']
            segment = result.best_ind[pos: pos + free_size]
            pos += free_size

            print(f"  Nodo '{spec['name']}' ({spec['kind']}, {free_size} parámetros libres):")

            if spec['kind'] == 'chance':
                reshaped = segment.reshape(spec['shape'])
                probs = softmax_rows(reshaped)
                print(f"    Probabilidades: {np.round(probs.flatten(), 3).tolist()}")

            elif spec['kind'] == 'utility':
                free_vals = apply_sigmoid(segment) * (self.util_max - self.util_min) + self.util_min
                print(f"    Utilidades asignadas: {np.round(free_vals, 2).tolist()}")
        print("")

    # ──────────────────────────────────────────────────────────────────────────
    # Guardar modelo
    # ──────────────────────────────────────────────────────────────────────────

    def save_model(self, output_path: str):
        """Guarda el modelo reconstruido en formato .xdsl."""
        self.net.write_file(output_path)
        print(f"[+] Modelo guardado en: {output_path}")

    # ──────────────────────────────────────────────────────────────────────────
    # Evaluación
    # ──────────────────────────────────────────────────────────────────────────

    def evaluate_rule(self, rule_idx: int) -> bool:
        """Evalúa una regla específica con los parámetros actuales."""
        evidence, target_node, expected_action = self.rules[rule_idx]
        try:
            res = self.engine.evaluate(self.nodes, evidence, self.net)
            actual = res.optimal_decisions.get(target_node)
            print(f"  Regla {rule_idx}: evidencia={evidence}")
            print(f"    Esperado: {target_node}={expected_action}")
            print(f"    Obtenido: {target_node}={actual}")
            print(f"    {'✓ SATISFECHA' if actual == expected_action else '✗ NO SATISFECHA'}")
            return actual == expected_action
        except Exception as e:
            print(f"  Regla {rule_idx}: ERROR en inferencia — {e}")
            return False

    def evaluate_all_rules(self):
        """Evalúa todas las reglas con los parámetros actuales e imprime un resumen."""
        print("\n[EVALUACIÓN COMPLETA DE REGLAS]")
        correct = 0
        for i in range(len(self.rules)):
            if self.evaluate_rule(i):
                correct += 1
        print(f"\nResultado: {correct}/{len(self.rules)} reglas satisfechas ({correct/len(self.rules):.1%})")
    
    # ──────────────────────────────────────────────────────────────────────────
    # Multi-start
    # ──────────────────────────────────────────────────────────────────────────

    def run_multistart(self, n_starts: int = 5, size_gen: int = 80, max_iter: int = 30, dead_iter: int = 10, log_csv: str = "convergencia_umda_ms.csv"):
        """Ejecuta varias corridas desde cero para evitar óptimos locales."""
        best_result = None
        best_cost = np.inf
        print(f"\n[*] Iniciando Multi-start: {n_starts} reinicios")

        for i in range(n_starts):
            print(f" ─> Reinicio {i+1}/{n_starts}")
            # Silenciamos la salida y no exportamos CSV en las pasadas intermedias
            result = self.run(size_gen=size_gen, max_iter=max_iter, dead_iter=dead_iter, verbose=False, log_csv=None)
            if result.best_cost < best_cost:
                best_cost = result.best_cost
                best_result = result
                print(f"    [✓] Nuevo mejor encontrado! Pérdida: {best_cost:.4f}")

        # Inyectamos los ganadores al modelo real
        self._vector_to_nodes(best_result.best_ind)
        self._print_diagnostics(best_result)

        # Exportamos la historia del mejor resultado encontrado al CSV
        if log_csv:
            with open(log_csv, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['generacion', 'mejor_coste', 'precision'])
                for i, cost in enumerate(best_result.history):
                    writer.writerow([i + 1, cost, 1.0 - cost])
            print(f"\n[+] Historial de convergencia del mejor reinicio guardado en: {log_csv}")

        return best_result

# ─────────────────────────────────────────────────────────────────────────────
# Punto de entrada
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    XDSL_PATH = "example/network-bypass2.xdsl"
    RULES_CSV = "example/reglas_generadas.csv"

    BEST_CONFIG = {"LIFEQ": "LIVE2AHQ", "ECONOMICALC": "LOW"}
    WORST_CONFIG = {"LIFEQ": "DEAD", "ECONOMICALC": "HIGH"}

    print("=== UMDAc continuo (Softmax/Sigmoide) ===")
    rec = IDRecovery(
        xdsl_path=XDSL_PATH,
        rules_csv=RULES_CSV,
        best_util_config=BEST_CONFIG,
        worst_util_config=WORST_CONFIG,
        util_range=(0.0, 10.0),
    )
    
    # Puedes usar run() directo o run_multistart() si la red es muy compleja
    result = rec.run(size_gen=100, max_iter=50, dead_iter=15, log_csv="convergencia_umda.csv")
    
    rec.evaluate_all_rules()
    rec.save_model("recovery_umda.xdsl")