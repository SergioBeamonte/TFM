"""
IDReconstructor - Improved version
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

# EDAspy imports - se usan los dos EDAs para comparar
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
    Usa el algoritmo de Kahn sobre el grafo de dependencias implícito en .parents.
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


def sigmoid_to_prob(x: np.ndarray) -> np.ndarray:
    """
    Transforma reales en (-inf, +inf) a probabilidades en (0, 1).
    Alternativa a clip: no hay sesgo por recorte.
    Útil si se usa UMDAc sin restricción de dominio.
    """
    return 1.0 / (1.0 + np.exp(-x))


# ─────────────────────────────────────────────────────────────────────────────
# Clase principal mejorada
# ─────────────────────────────────────────────────────────────────────────────

class IDReconstructor:
    """
    Reconstruye los parámetros de un Diagrama de Influencia de estructura fija
    a partir de un conjunto de reglas expertas, usando EDAs.

    Parámetros
    ----------
    xdsl_path : str
        Ruta al fichero .xdsl con la estructura del ID (parámetros iniciales irrelevantes).
    rules_csv : str
        CSV con las reglas. Cada fila es una regla; columnas = nodos.
        Valores: 0 = no en la regla, +k = evidencia (estado k-1), -k = decisión esperada.
    best_util_config : dict
        Configuración de padres del nodo utilidad que fija el valor máximo.
    worst_util_config : dict
        Configuración de padres del nodo utilidad que fija el valor mínimo.
    rule_weights : list[float], opcional
        Peso de cada regla. Si None, todos los pesos son 1.
    use_sigmoid : bool
        Si True, el vector del EDA vive en ℝ y se transforma con sigmoid.
        Si False, el vector vive en [0,1] y se recorta (con posible sesgo).
    util_range : tuple
        (min_util, max_util) — rango de valores de utilidad. Por defecto (0, 10).
    """

    def __init__(
        self,
        xdsl_path: str,
        rules_csv: str,
        best_util_config: Dict[str, str],
        worst_util_config: Dict[str, str],
        rule_weights: Optional[List[float]] = None,
        use_sigmoid: bool = True,
        util_range: Tuple[float, float] = (0.0, 10.0),
    ):
        self.xdsl_path = xdsl_path
        self.rules_csv = rules_csv
        self.best_util_config = best_util_config
        self.worst_util_config = worst_util_config
        self.use_sigmoid = use_sigmoid
        self.util_min, self.util_max = util_range

        # 1. Cargar modelo
        self.net = pysmile.Network()
        self.net.read_file(xdsl_path)
        self.nodes = NetworkExtractor.extract(self.net)

        # 2. Parsear reglas
        self.rules = self._parse_rules(rules_csv)
        if rule_weights is not None:
            assert len(rule_weights) == len(self.rules), \
                "rule_weights debe tener la misma longitud que el número de reglas."
            self.rule_weights = np.array(rule_weights, dtype=float)
        else:
            self.rule_weights = np.ones(len(self.rules), dtype=float)
        self.rule_weights /= self.rule_weights.sum()  # normalizar

        # 3. Ordenación topológica (padres antes que hijos)
        self.topo_order = topological_sort(self.nodes)

        # 4. Especificaciones de parámetros libres (excluye entradas fijas de utilidad)
        self.param_specs = self._get_param_specs()
        self.total_vars = sum(spec['free_size'] for spec in self.param_specs)

        # 5. Motor de inferencia (se instancia una sola vez)
        self.engine = ShachterEngine()

        print(f"[+] Reglas cargadas:           {len(self.rules)}")
        print(f"[+] Nodos en orden topológico: {self.topo_order}")
        print(f"[+] Variables libres a opti.:  {self.total_vars}")
        if use_sigmoid:
            print("[+] Modo: sigmoid (vector en ℝ, sin sesgo por recorte)")
        else:
            print("[+] Modo: clip [0,1] (más rápido, posible sesgo leve)")

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
                    # FIX BUG 3: parseo robusto
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
                idx = self._get_table_idx(nd, config)
                fixed.append(idx)
            except (KeyError, ValueError):
                pass  # config no aplica a este nodo de utilidad
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
                    'size': nd.table.size,       # total de entradas en la tabla
                    'free_size': nd.table.size,  # todas libres para chance
                    'shape': nd.table.shape,
                    'fixed_indices': [],
                    'fixed_values': [],
                })

            elif nd.kind == NodeKind.UTILITY:
                # FIX BUG 2: identificar y excluir entradas fijas
                fixed_indices = self._get_fixed_util_indices(nd)
                fixed_values = []
                flat_fixed_positions = set()

                reshaped = np.zeros(nd.table.shape)
                for idx in fixed_indices:
                    is_best = (idx == self._get_table_idx(nd, self.best_util_config)
                               if self.best_util_config else False)
                    val = self.util_max if is_best else self.util_min
                    fixed_values.append((idx, val))
                    # Calcular posición plana
                    flat_pos = np.ravel_multi_index(idx, nd.table.shape)
                    flat_fixed_positions.add(flat_pos)

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

    def _get_table_idx(self, nd, config: Dict[str, str]) -> tuple:
        """Convierte un diccionario de estados padre en un índice numpy multidimensional."""
        idx = []
        for p_name in nd.parents:
            state = config[p_name]
            idx.append(self.nodes[p_name].states.index(state))
        return tuple(idx)

    # ──────────────────────────────────────────────────────────────────────────
    # Traducción vector → parámetros del modelo
    # ──────────────────────────────────────────────────────────────────────────

    def _vector_to_nodes(self, vector: np.ndarray):
        """
        Traduce el vector del EDA en parámetros concretos y los aplica
        tanto a self.nodes como a self.net (pysmile).

        FIX BUG 1: ahora sincroniza self.net con los parámetros actualizados.
        FIX BUG 4: manejo correcto de nodos raíz (tabla 1D).
        FIX BUG 6: soporte para sigmoid (sin sesgo por recorte).
        """
        pos = 0
        for spec in self.param_specs:
            name = spec['name']
            nd = self.nodes[name]
            free_size = spec['free_size']
            raw = vector[pos: pos + free_size].copy()
            pos += free_size

            if spec['kind'] == 'chance':
                # Transformar a probabilidades
                if self.use_sigmoid:
                    probs = sigmoid_to_prob(raw)  # sin sesgo por recorte
                else:
                    probs = np.clip(raw, 1e-6, 1.0 - 1e-6)

                # FIX BUG 4: reshape seguro para nodos raíz (1D) y nodos con padres (nD)
                reshaped = probs.reshape(spec['shape'])

                # Normalización por filas (última dimensión = estados propios)
                # softmax garantiza >0 y suma=1 sin división por cero
                normalized = softmax_rows(reshaped)
                nd.table = normalized

                # FIX BUG 1: sincronizar con pysmile
                h = self.net.get_node(name)
                self.net.set_node_definition(h, normalized.flatten().tolist())

            elif spec['kind'] == 'utility':
                # Reconstruir tabla completa desde los valores libres
                full_flat = np.zeros(spec['size'])
                # Colocar valores libres (escalados al rango de utilidad)
                if self.use_sigmoid:
                    free_vals = sigmoid_to_prob(raw) * (self.util_max - self.util_min) + self.util_min
                else:
                    free_vals = np.clip(raw, 0.0, 1.0) * (self.util_max - self.util_min) + self.util_min

                full_flat[spec['free_mask']] = free_vals

                # Colocar valores fijos (best/worst)
                reshaped = full_flat.reshape(spec['shape'])
                for idx, val in spec['fixed_values']:
                    reshaped[idx] = val
                nd.table = reshaped

                # FIX BUG 1: sincronizar con pysmile
                h = self.net.get_node(name)
                self.net.set_node_definition(h, reshaped.flatten().tolist())

    # ──────────────────────────────────────────────────────────────────────────
    # Función de aptitud
    # ──────────────────────────────────────────────────────────────────────────

    def fitness(self, solution: np.ndarray) -> float:
        """
        Evalúa un candidato. Devuelve coste ∈ [0, 1] (minimizar).
        FIX BUG 5: engine se reutiliza (instanciado en __init__).
        """
        self._vector_to_nodes(solution)

        weighted_correct = 0.0
        for (evidence, target_node, expected_action), weight in zip(self.rules, self.rule_weights):
            try:
                res = self.engine.evaluate(self.nodes, evidence, self.net)
                if res.optimal_decisions.get(target_node) == expected_action:
                    weighted_correct += weight
            except Exception as e:
                # Fallo de inferencia: regla no satisfecha, penalización implícita
                pass

        return 1.0 - weighted_correct  # minimizar

    # ──────────────────────────────────────────────────────────────────────────
    # Ejecución del EDA
    # ──────────────────────────────────────────────────────────────────────────

    def _build_initial_mean(self) -> np.ndarray:
        """
        Construye la media inicial del modelo EDA usando el orden topológico:
        - Nodos raíz: uniforme (sin información previa).
        - Nodos con padres: uniforme también, pero el orden garantiza que
          la inicialización respeta la estructura del grafo.
        - En modo sigmoid, 0.0 → sigmoid(0.0) = 0.5 → correcto.
        - En modo clip, 0.5 → uniforme → correcto.
        """
        # En ambos modos, inicializar en 0.0 (sigmoid→0.5) o 0.5 (clip→0.5)
        if self.use_sigmoid:
            return np.zeros(self.total_vars)
        else:
            return np.full(self.total_vars, 0.5)

    def run(
        self,
        size_gen: int = 100,
        max_iter: int = 50,
        dead_iter: int = 15,
        alpha: float = 0.5,
        verbose: bool = True,
    ):
        """
        Ejecuta UMDAc para encontrar los parámetros que satisfacen las reglas.

        Parámetros
        ----------
        size_gen : int
            Tamaño de la población por generación. Recomendado ≥ 100 para
            d < 100 variables libres.
        max_iter : int
            Número máximo de generaciones.
        dead_iter : int
            Parar si no hay mejora en este número de generaciones.
            Aumentar si el problema tiene muchas variables libres.
        alpha : float
            Fracción de selección (τ). 0.5 = selecciona la mitad mejor.
        verbose : bool
            Mostrar progreso por generación.
        """
        # Configurar UMDAc
        # Si usamos sigmoid, el dominio es ℝ (sin bounds estrictos)
        # Si usamos clip, el dominio es [0, 1]
        if self.use_sigmoid:
            lower, upper = -6.0, 6.0  # sigmoid(-6)≈0.002, sigmoid(6)≈0.998
        else:
            lower, upper = 0.0, 1.0

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
            print(f"    Reglas: {len(self.rules)} (pesos: {self.rule_weights.round(3)})")

        result = umda.minimize(self.fitness, verbose=verbose)

        accuracy = 1.0 - result.best_cost
        print(f"\n{'─'*50}")
        print(f"  Optimización finalizada")
        print(f"  Precisión final: {accuracy:.1%} ({int(accuracy * len(self.rules))}/{len(self.rules)} reglas)")
        print(f"  Generaciones ejecutadas: {result.n_iter}")
        print(f"{'─'*50}")

        # Aplicar la mejor solución
        self._vector_to_nodes(result.best_ind)

        # Diagnóstico del modelo final
        self._print_diagnostics(result)

        return result

    # ──────────────────────────────────────────────────────────────────────────
    # Diagnóstico
    # ──────────────────────────────────────────────────────────────────────────

    def _print_diagnostics(self, result):
        """
        Muestra qué parámetros están bien determinados por las reglas
        (varianza baja en el modelo final) y cuáles no (varianza alta).
        Esta información tiene valor directo para el experto.
        """
        print("\n[DIAGNÓSTICO] Determinación de parámetros por las reglas:")
        print("  Alta varianza → reglas no determinan este parámetro (libre)")
        print("  Baja varianza → reglas determinan bien este parámetro\n")

        pos = 0
        for spec in self.param_specs:
            free_size = spec['free_size']
            # La varianza del modelo final está en result.model_std o similar
            # Si UMDAc no la expone directamente, usamos la dispersión del best_ind
            # como proxy (no ideal, pero informativo)
            segment = result.best_ind[pos: pos + free_size]
            if self.use_sigmoid:
                probs = sigmoid_to_prob(segment)
            else:
                probs = np.clip(segment, 0, 1)

            print(f"  Nodo '{spec['name']}' ({spec['kind']}, {free_size} parámetros libres):")
            print(f"    Valores óptimos (redondeados a 2 dec.): {np.round(probs, 2)}")
            pos += free_size

    # ──────────────────────────────────────────────────────────────────────────
    # Guardar modelo
    # ──────────────────────────────────────────────────────────────────────────

    def save_model(self, output_path: str):
        """Guarda el modelo reconstruido en formato .xdsl."""
        # Los parámetros ya fueron sincronizados con self.net en _vector_to_nodes
        self.net.write_file(output_path)
        print(f"[+] Modelo guardado en: {output_path}")

    # ──────────────────────────────────────────────────────────────────────────
    # Evaluación de una sola regla (útil para debugging)
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


# ─────────────────────────────────────────────────────────────────────────────
# ALTERNATIVA 1: UMDAcat (discreto puro)
# ─────────────────────────────────────────────────────────────────────────────

class IDReconstructorDiscrete(IDReconstructor):
    """
    Variante que usa un modelo EDA puramente discreto (UMDAcat).

    Discretiza cada parámetro de CPT en L niveles uniformes:
        {0/L, 1/L, ..., L/L}
    y trata cada nivel como un valor categórico. Más apropiado cuando
    se quiere interpretar el modelo probabilístico del EDA en términos
    de distribuciones sobre valores discretos de probabilidad.

    Requiere una implementación manual de UMDAcat porque EDAspy no tiene
    UMDAcat con dominios de tamaño variable por dimensión.
    """

    def __init__(self, *args, L: int = 10, **kwargs):
        """
        L : int
            Número de niveles de discretización. L=10 da resolución 0.1.
            L=20 da resolución 0.05. Aumentar L mejora la precisión pero
            aumenta el espacio de búsqueda.
        """
        # Forzar modo no-sigmoid (trabajamos con enteros)
        kwargs['use_sigmoid'] = False
        super().__init__(*args, **kwargs)
        self.L = L

        # Redefinir total_vars en términos discretos
        # Cada parámetro libre es ahora un entero en {0, 1, ..., L}
        print(f"[+] Modo discreto: L={L} niveles por parámetro")
        print(f"    Cada CPT row se muestrea de {L+1} valores posibles")

    def _sample_population(self, model_probs: List[np.ndarray], size: int) -> np.ndarray:
        """
        Muestrea 'size' individuos del modelo probabilístico.
        model_probs[k] es un array de forma (L+1,) con la distribución
        sobre los L+1 valores del parámetro k.
        """
        pop = np.zeros((size, self.total_vars), dtype=int)
        for k, probs in enumerate(model_probs):
            pop[:, k] = np.random.choice(len(probs), size=size, p=probs)
        return pop

    def _int_to_prob(self, val: int) -> float:
        """Convierte un entero en {0,...,L} a probabilidad en [0,1]."""
        return val / self.L

    def _vector_to_nodes_discrete(self, int_vector: np.ndarray):
        """Versión discreta de _vector_to_nodes: acepta enteros en {0,...,L}."""
        prob_vector = int_vector.astype(float) / self.L
        self._vector_to_nodes(prob_vector)

    def fitness_discrete(self, int_vector: np.ndarray) -> float:
        """Función de aptitud para el vector discreto."""
        self._vector_to_nodes_discrete(int_vector)
        return self._evaluate_rules()

    def _evaluate_rules(self) -> float:
        """Lógica de evaluación separada para reutilización."""
        weighted_correct = 0.0
        for (evidence, target_node, expected_action), weight in zip(self.rules, self.rule_weights):
            try:
                res = self.engine.evaluate(self.nodes, evidence, self.net)
                if res.optimal_decisions.get(target_node) == expected_action:
                    weighted_correct += weight
            except Exception:
                pass
        return 1.0 - weighted_correct

    def run_discrete(
        self,
        size_gen: int = 100,
        max_iter: int = 50,
        dead_iter: int = 15,
        tau: float = 0.5,
        epsilon: float = 0.01,
        verbose: bool = True,
    ):
        """
        Implementación manual de UMDAcat para CPTs discretas.

        tau : float
            Fracción de selección.
        epsilon : float
            Suavizado mínimo de probabilidades (evita colapso prematuro).
        """
        d = self.total_vars
        n_vals = self.L + 1  # Valores posibles por parámetro: 0, 1, ..., L

        # Modelo inicial: uniforme sobre todos los valores
        model = [np.ones(n_vals) / n_vals for _ in range(d)]

        best_cost = np.inf
        best_ind = None
        no_improve = 0

        print(f"\n[*] UMDAcat discreto: L={self.L}, d={d}, gen={size_gen}, iter={max_iter}")

        for t in range(max_iter):
            # 1. Muestreo
            pop = self._sample_population(model, size_gen)

            # 2. Evaluación
            costs = np.array([self.fitness_discrete(pop[j]) for j in range(size_gen)])

            # 3. Selección (mejores K individuos)
            K = max(1, int(tau * size_gen))
            selected_idx = np.argsort(costs)[:K]
            selected = pop[selected_idx]
            best_gen_cost = costs[selected_idx[0]]

            # 4. Actualización del modelo (frecuencias empíricas + suavizado)
            new_model = []
            for k in range(d):
                freq = np.zeros(n_vals)
                for val in selected[:, k]:
                    freq[val] += 1
                freq /= K
                # Suavizado para evitar colapso
                freq = (1 - epsilon) * freq + epsilon / n_vals
                new_model.append(freq)
            model = new_model

            # 5. Seguimiento del mejor global
            if best_gen_cost < best_cost:
                best_cost = best_gen_cost
                best_ind = pop[selected_idx[0]].copy()
                no_improve = 0
            else:
                no_improve += 1

            if verbose:
                acc = 1.0 - best_cost
                print(f"  Gen {t+1:3d}: mejor_coste={best_cost:.4f} | precisión={acc:.1%} | no_mejora={no_improve}")

            if no_improve >= dead_iter:
                print(f"\n[!] Parada temprana: sin mejora en {dead_iter} generaciones.")
                break

        accuracy = 1.0 - best_cost
        print(f"\n{'─'*50}")
        print(f"  UMDAcat finalizado")
        print(f"  Precisión final: {accuracy:.1%} ({int(accuracy * len(self.rules))}/{len(self.rules)} reglas)")
        print(f"{'─'*50}")

        # Aplicar mejor solución
        self._vector_to_nodes_discrete(best_ind)

        # Devolver modelo final (distribuciones aprendidas)
        return best_ind, best_cost, model


# ─────────────────────────────────────────────────────────────────────────────
# ALTERNATIVA 2: Reinicio con perturbación (escapar óptimos locales)
# ─────────────────────────────────────────────────────────────────────────────

class IDReconstructorMultiStart(IDReconstructor):
    """
    Ejecuta múltiples corridas de UMDAc con inicializaciones diferentes
    y devuelve la mejor solución encontrada.

    Útil cuando la función de aptitud es multimodal (hay varias asignaciones
    de parámetros que satisfacen subconjuntos diferentes de las reglas).
    """

    def run_multistart(
        self,
        n_starts: int = 5,
        size_gen: int = 80,
        max_iter: int = 30,
        dead_iter: int = 10,
        verbose: bool = False,
    ):
        """
        n_starts : int
            Número de reinicializaciones independientes.
        """
        best_result = None
        best_cost = np.inf

        print(f"\n[*] Multi-start: {n_starts} reinicializaciones")

        for i in range(n_starts):
            print(f"\n  --- Inicio {i+1}/{n_starts} ---")
            result = self.run(
                size_gen=size_gen,
                max_iter=max_iter,
                dead_iter=dead_iter,
                verbose=verbose,
            )
            if result.best_cost < best_cost:
                best_cost = result.best_cost
                best_result = result
                print(f"  [✓] Nuevo mejor: {1.0 - best_cost:.1%}")

        print(f"\n{'═'*50}")
        print(f"  Multi-start finalizado")
        print(f"  Mejor precisión global: {1.0 - best_cost:.1%}")
        print(f"{'═'*50}")

        # Aplicar el mejor resultado global
        self._vector_to_nodes(best_result.best_ind)
        return best_result


# ─────────────────────────────────────────────────────────────────────────────
# Punto de entrada
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    XDSL_PATH = "copia_modelo.xdsl"
    RULES_CSV = "reglas_ejemplo.csv"

    BEST_CONFIG = {"LIFEQ": "LIVE2AHQ", "ECONOMICALC": "LOW"}
    WORST_CONFIG = {"LIFEQ": "DEAD", "ECONOMICALC": "HIGH"}

    # ── Opción 1: UMDAc continuo (recomendado para empezar) ──
    print("=== OPCIÓN 1: UMDAc continuo con sigmoid ===")
    rec = IDReconstructor(
        XDSL_PATH, RULES_CSV,
        BEST_CONFIG, WORST_CONFIG,
        use_sigmoid=True,        # sin sesgo por recorte
        util_range=(0.0, 10.0),
    )
    result = rec.run(size_gen=100, max_iter=50, dead_iter=15)
    rec.evaluate_all_rules()
    rec.save_model("reconstructed_umda.xdsl")

    # ── Opción 2: UMDAcat discreto (más interpretable) ──
    print("\n=== OPCIÓN 2: UMDAcat discreto (L=10) ===")
    rec_disc = IDReconstructorDiscrete(
        XDSL_PATH, RULES_CSV,
        BEST_CONFIG, WORST_CONFIG,
        L=10,
    )
    best_ind, best_cost, final_model = rec_disc.run_discrete(
        size_gen=100, max_iter=50, dead_iter=15, epsilon=0.01
    )
    rec_disc.evaluate_all_rules()
    rec_disc.save_model("reconstructed_umda_cat.xdsl")

    # ── Opción 3: Multi-start (para escapar óptimos locales) ──
    print("\n=== OPCIÓN 3: Multi-start (5 reinicializaciones) ===")
    rec_ms = IDReconstructorMultiStart(
        XDSL_PATH, RULES_CSV,
        BEST_CONFIG, WORST_CONFIG,
        use_sigmoid=True,
    )
    result_ms = rec_ms.run_multistart(n_starts=5, size_gen=80, max_iter=30)
    rec_ms.save_model("reconstructed_multistart.xdsl")