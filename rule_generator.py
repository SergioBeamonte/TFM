import csv
import random
import os
import numpy as np
import pysmile
import pysmile_license
from typing import Dict, List, Optional
from extractor import NetworkExtractor
from engines import ShachterEngine, BaseEngine
from models import IDNode, NodeKind, EvaluationResult

class RuleGenerator:
    """
    Generador de reglas basado en un Diagrama de Influencia.
    Genera escenarios aleatorios y encuentra la decisión óptima para crear reglas IF-THEN codificadas en CSV.
    """
    def __init__(self, xdsl_path: str):
        if not os.path.exists(xdsl_path):
            raise FileNotFoundError(f"No se encontró el archivo: {xdsl_path}")
        
        self.net = pysmile.Network()
        self.net.read_file(xdsl_path)
        self.nodes = NetworkExtractor.extract(self.net)
        
    def generate_csv(self, n_rules: int, output_path: str, fixed_states: Dict[str, str] = None, respect_probs: bool = True):
        """
        Genera un archivo CSV con n_rules, siguiendo la codificación especificada.
        """
        fixed_states = fixed_states or {}
        
        # Identificar nodos por tipo
        chance_nodes = [n.name for n in self.nodes.values() if n.kind == NodeKind.CHANCE]
        decision_nodes = [n.name for n in self.nodes.values() if n.kind == NodeKind.DECISION]
        
        # Orden de columnas: Azar primero, luego Decisión
        node_order = sorted(chance_nodes) + sorted(decision_nodes)
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(node_order)
            
            for i in range(n_rules):
                row = self._generate_single_rule(node_order, fixed_states, respect_probs)
                writer.writerow(row)
        
        print(f"[+] Generadas {n_rules} reglas en: {output_path}")

    def export_mappings(self, output_path: str = "mappings.txt"):
        """
        Exporta la equivalencia entre los índices numéricos y los nombres de los estados en formato TXT (diccionario).
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("EQUIVALENCIAS DE CODIFICACION CSV\n")
            f.write("=================================\n\n")
            f.write("Para nodos de azar y de entrada (predecesores):\n")
            f.write("  Valor N > 0  => Estado de índice N-1\n")
            f.write("Para el nodo de decisión objetivo:\n")
            f.write("  Valor N < 0  => Acción de índice |N|-1\n\n")
            f.write("DETALLE POR NODO:\n")
            f.write("-----------------\n")
            
            for name, nd in sorted(self.nodes.items()):
                if nd.kind == NodeKind.UTILITY:
                    continue
                f.write(f"\n[{name}] ({nd.kind.name})\n")
                for i, state in enumerate(nd.states):
                    f.write(f"  {i+1} : {state}\n")
        print(f"[+] Mapeos exportados a: {output_path}")

    def _generate_single_rule(self, node_order: List[str], fixed_states: Dict[str, str], respect_probs: bool) -> List[int]:
        # 1. Muestreo de todos los nodos de azar (top-down)
        sampled_chance = self._sample_chance_nodes(fixed_states, respect_probs)
        
        # 2. Elegir un nodo de decisión objetivo al azar
        dn_names = [n.name for n in self.nodes.values() if n.kind == NodeKind.DECISION]
        target_dn_name = random.choice(dn_names)
        
        # 3. Evaluar para obtener decisiones óptimas
        # Nota: ShachterEngine evalúa todas las decisiones de forma secuencial
        engine = ShachterEngine()
        res = engine.evaluate(self.nodes, sampled_chance, self.net)
        
        # 4. Construir la fila según la codificación solicitada
        row = []
        for name in node_order:
            nd = self.nodes[name]
            
            if name == target_dn_name:
                # Nodo decisor objetivo: valor negativo -(index+1)
                decision_str = res.optimal_decisions[name]
                action_idx = nd.states.index(decision_str)
                row.append(-(action_idx + 1))
                
            elif self._is_predecessor(name, target_dn_name):
                # Nodo que interfiere (input): valor positivo (index+1)
                # Si es azar, usamos el valor muestreado. Si es decisión, el valor óptimo del motor.
                val_str = sampled_chance.get(name) or res.optimal_decisions.get(name)
                val_idx = nd.states.index(val_str)
                row.append(val_idx + 1)
                
            else:
                # No interfiere en esta regla específica
                row.append(0)
                
        return row

    def _sample_chance_nodes(self, fixed_states: Dict[str, str], respect_probs: bool) -> Dict[str, str]:
        """Muestreo descendente de los nodos de azar respetando dependencias."""
        # Filtramos nodos de utilidad para el orden topológico
        eval_nodes = {k: v for k, v in self.nodes.items() if v.kind != NodeKind.UTILITY}
        order = BaseEngine._topo_sort(eval_nodes)
        
        sampled = {}
        for name in order:
            nd = self.nodes[name]
            
            # Solo muestreamos azar. Los de decisión se ignoran aquí (se evalúan luego).
            if nd.kind != NodeKind.CHANCE:
                continue
                
            if name in fixed_states:
                sampled[name] = fixed_states[name]
                continue
            
            if not nd.parents:
                # NODO RAÍZ (Sin padres)
                if not respect_probs:
                    # Muestreo uniforme, pero excluyendo los que tienen 0% de probabilidad base
                    valid_states = [s for i, s in enumerate(nd.states) if nd.table[i] > 0]
                    # Fallback por si la tabla está rota y todos son 0
                    sampled[name] = random.choice(valid_states if valid_states else nd.states)
                else:
                    # Muestreo probabilístico ponderado
                    p = np.array(nd.table)
                    total = np.sum(p)
                    p = p / total if total > 0 else np.ones(len(nd.states)) / len(nd.states)
                    sampled[name] = np.random.choice(nd.states, p=p)
            else:
                # NODO INTERMEDIO: Siempre navegamos el CPT para obtener el contexto de los padres
                stride = 1
                offset = 0
                for p_name in reversed(nd.parents):
                    p_val = sampled[p_name] 
                    p_states = self.nodes[p_name].states
                    p_idx = p_states.index(p_val)
                    offset += p_idx * stride
                    stride *= len(p_states)
                
                num_states = len(nd.states)
                start = offset * num_states
                probs = np.array(nd.table[start : start + num_states])
                
                if not respect_probs:
                    # NUEVA LÓGICA: Muestreo uniforme PERO solo sobre estados lógicamente posibles
                    valid_indices = np.where(probs > 0)[0]
                    
                    if len(valid_indices) > 0:
                        valid_states = [nd.states[i] for i in valid_indices]
                        sampled[name] = random.choice(valid_states)
                    else:
                        # Fallback de seguridad extrema: si los estados de los padres han
                        # derivado en una fila del CPT completamente a ceros (escenario imposible).
                        sampled[name] = random.choice(nd.states)
                else:
                    # Lógica original: Muestreo probabilístico ponderado
                    total = np.sum(probs)
                    if total > 0:
                        probs = probs / total
                    else:
                        probs = np.ones(num_states) / num_states
                    
                    sampled[name] = np.random.choice(nd.states, p=probs)
                    
        return sampled


    # Lógica alternativa para muestreo de nodos de azar (más robusta y bayesiana)

    # def _sample_chance_nodes(self, fixed_states: Dict[str, str], respect_probs: bool) -> Dict[str, str]:
    #     """
    #     Muestreo bayesiano real: Inyecta evidencia, propaga la información 
    #     hacia arriba/abajo, y luego muestrea dinámicamente.
    #     """
    #     # Asegurarnos de que el motor está limpio de iteraciones anteriores
    #     self.net.clear_all_evidence()
        
    #     # 1. Inyectar la "realidad forzada" (fixed_states) como evidencia dura
    #     for name, state in fixed_states.items():
    #         if name in self.nodes and self.nodes[name].kind == NodeKind.CHANCE:
    #             # Le decimos a SMILE: "Esto es un hecho inamovible"
    #             self.net.set_evidence(name, state)
                
    #     # Obtenemos el orden topológico
    #     eval_nodes = {k: v for k, v in self.nodes.items() if v.kind != NodeKind.UTILITY}
    #     order = BaseEngine._topo_sort(eval_nodes)
        
    #     sampled = {}
    #     for name in order:
    #         nd = self.nodes[name]
            
    #         if nd.kind != NodeKind.CHANCE:
    #             continue
                
    #         # Si el nodo es de los que el usuario ya ha fijado, lo guardamos y avanzamos
    #         if name in fixed_states:
    #             sampled[name] = fixed_states[name]
    #             continue
                
    #         # 2. LA MAGIA BAYESIANA: Recalcular TODO el grafo
    #         # Esto obliga a SMILE a propagar la evidencia (los fixed_states y lo que 
    #         # ya hayamos muestreado) para recalcular la probabilidad exacta de ESTE nodo.
    #         try:
    #             self.net.update_beliefs()
    #             # get_node_value devuelve el array de probabilidades posteriores (Beliefs)
    #             beliefs = self.net.get_node_value(name)
    #             probs = np.array(beliefs)
    #         except pysmile.SMILEException:
    #             # Fallback: Si las evidencias forzadas entran en una paradoja lógica imposible 
    #             # según tu red (ej: forzar Embarazo=Sí en Hombre=Sí), SMILE lanza excepción.
    #             probs = np.ones(len(nd.states)) / len(nd.states)
                
    #         # Normalización estándar (por seguridad de numpy)
    #         total = np.sum(probs)
    #         if total > 0:
    #             probs = probs / total
    #         else:
    #             probs = np.ones(len(nd.states)) / len(nd.states)
                
    #         # 3. Muestreo sobre las probabilidades posteriores calculadas
    #         if respect_probs:
    #             # Muestreo ponderado normal
    #             chosen_state = np.random.choice(nd.states, p=probs)
    #         else:
    #             # Muestreo Uniforme Inteligente (Solo sobre lo que es lógicamente posible)
    #             valid_indices = np.where(probs > 0)[0]
    #             if len(valid_indices) > 0:
    #                 valid_states = [nd.states[i] for i in valid_indices]
    #                 chosen_state = random.choice(valid_states)
    #             else:
    #                 chosen_state = random.choice(nd.states)
                    
    #         sampled[name] = chosen_state
            
    #         # 4. CRÍTICO: El valor que acabamos de inventar se convierte en una 
    #         # nueva "realidad" (evidencia). Así condicionará correctamente a los 
    #         # nodos que evaluemos en las siguientes vueltas del bucle.
    #         self.net.set_evidence(name, chosen_state)
            
    #     # Limpiar la red antes de salir para no contaminar la siguiente fila del CSV
    #     self.net.clear_all_evidence()
    #     return sampled

    def _is_predecessor(self, potential_pred: str, target: str) -> bool:
        """Verifica si potential_pred es un antecesor (ancestro) de target en el DAG."""
        if potential_pred == target:
            return False
        
        stack = [target]
        visited = {target}
        while stack:
            curr = stack.pop()
            if curr == potential_pred:
                return True
            for p in self.nodes[curr].parents:
                if p not in visited:
                    visited.add(p)
                    stack.append(p)
        return False

if __name__ == "__main__":
    # Test rápido de generación
    try:
        gen = RuleGenerator(r"TFM\example\network-bypass2.xdsl")
        gen.generate_csv(n_rules=30, output_path=r"TFM\example\reglas_generadas.csv", respect_probs=False)
    except Exception as e:
        print(f"[!] Error: {e}")
