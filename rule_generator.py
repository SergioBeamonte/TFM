"""
RuleGenerator
===================================
"""

import csv
import random
import os
import numpy as np
import pysmile
import pysmile_license
from typing import Dict, List, Optional

from extractor import NetworkExtractor
from engines import ShachterEngine, BaseEngine
from models import NodeKind


class RuleGenerator:
    """
    Generador de reglas basado en un Diagrama de Influencia.
    Genera escenarios aleatorios y encuentra la decisión óptima para crear reglas IF-THEN codificadas en CSV.
    Garantiza que no haya filas (reglas) duplicadas.
    """
    def __init__(self, xdsl_path: str):
        if not os.path.exists(xdsl_path):
            raise FileNotFoundError(f"No se encontró el archivo: {xdsl_path}")
        
        self.net = pysmile.Network()
        self.net.read_file(xdsl_path)
        self.nodes = NetworkExtractor.extract(self.net)
        
    def generate_csv(self, n_rules: int, output_path: str, fixed_states: Dict[str, str] = None, respect_probs: bool = True):
        """
        Genera un archivo CSV con n_rules únicas.
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
            
            unique_rules = set()
            attempts = 0
            max_attempts = n_rules * 100  # Cortafuegos contra bucles infinitos
            
            while len(unique_rules) < n_rules and attempts < max_attempts:
                attempts += 1
                row = self._generate_single_rule(node_order, fixed_states, respect_probs)
                
                # Las listas no se pueden hashear en un set, la pasamos a tupla
                row_tuple = tuple(row)
                
                if row_tuple not in unique_rules:
                    unique_rules.add(row_tuple)
                    writer.writerow(row)
            
            if len(unique_rules) < n_rules:
                print(f"[!] AVISO: El modelo se ha quedado sin combinaciones únicas.")
                print(f"    Se pidieron {n_rules}, pero solo se generaron {len(unique_rules)} tras {attempts} intentos.")
            else:
                print(f"[+] Generadas {n_rules} reglas ÚNICAS en: {output_path} (Intentos totales: {attempts})")

    def export_mappings(self, output_path: str = "mappings.txt"):
        """
        Exporta la equivalencia entre los índices numéricos y los nombres de los estados en formato TXT.
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
        # 1. Muestreo de todos los nodos de azar
        sampled_chance = self._sample_chance_nodes(fixed_states, respect_probs)
        
        # 2. Elegir un nodo de decisión objetivo al azar
        dn_names = [n.name for n in self.nodes.values() if n.kind == NodeKind.DECISION]
        target_dn_name = random.choice(dn_names)
        
        # 3. Evaluar para obtener decisiones óptimas
        engine = ShachterEngine()
        res = engine.evaluate(self.nodes, sampled_chance, self.net)
        
        # 4. Construir la fila
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
                val_str = sampled_chance.get(name) or res.optimal_decisions.get(name)
                val_idx = nd.states.index(val_str)
                row.append(val_idx + 1)
                
            else:
                # No interfiere
                row.append(0)
                
        return row

    def _sample_chance_nodes(self, fixed_states: Dict[str, str], respect_probs: bool) -> Dict[str, str]:
        """Muestreo descendente de los nodos de azar y decisiones previas respetando dependencias."""
        eval_nodes = {k: v for k, v in self.nodes.items() if v.kind != NodeKind.UTILITY}
        order = BaseEngine._topo_sort(eval_nodes)
        
        sampled = {}
        for name in order:
            nd = self.nodes[name]
            
            # Si el usuario lo ha fijado manualmente, lo respetamos
            if name in fixed_states:
                sampled[name] = fixed_states[name]
                continue
            
            # Si es un nodo de decisión, debemos simular que "se tomó una decisión" 
            # de forma aleatoria para que los nodos de azar hijos puedan leer su tabla.
            if nd.kind == NodeKind.DECISION:
                sampled[name] = random.choice(nd.states)
                continue
            
            if nd.kind != NodeKind.CHANCE:
                continue
            
            # Nos aseguramos de que la tabla sea siempre de 1 sola dimensión
            flat_table = np.array(nd.table).flatten()
            
            if not nd.parents:
                # NODO RAÍZ
                if not respect_probs:
                    valid_states = [s for i, s in enumerate(nd.states) if flat_table[i] > 0]
                    sampled[name] = random.choice(valid_states if valid_states else nd.states)
                else:
                    p = flat_table
                    total = np.sum(p)
                    p = p / total if total > 0 else np.ones(len(nd.states)) / len(nd.states)
                    sampled[name] = np.random.choice(nd.states, p=p)
            else:
                # NODO INTERMEDIO
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
                
                # Leemos los datos exclusivamente de la versión aplanada (1D)
                probs = flat_table[start : start + num_states]
                
                if not respect_probs:
                    valid_indices = np.where(probs > 0)[0]
                    if len(valid_indices) > 0:
                        valid_states = [nd.states[i] for i in valid_indices]
                        sampled[name] = random.choice(valid_states)
                    else:
                        sampled[name] = random.choice(nd.states)
                else:
                    total = np.sum(probs)
                    if total > 0:
                        probs = probs / total
                    else:
                        probs = np.ones(num_states) / num_states
                    
                    # np.random.choice ahora recibe siempre un array 1D
                    sampled[name] = np.random.choice(nd.states, p=probs)
                    
        return sampled

    def _is_predecessor(self, potential_pred: str, target: str) -> bool:
        """Verifica si potential_pred es un antecesor de target en el DAG."""
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

    def export_mappings(self, output_path: str = "mappings.txt"):
        """
        Genera un archivo de texto con la leyenda de estados y números.
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("LEYENDA DE CODIFICACIÓN PARA REGLAS CSV\n")
            f.write("========================================\n\n")
            f.write("REGLA GENERAL:\n")
            f.write("1. VALORES POSITIVOS (>0): Nodo de Evidencia (Input).\n")
            f.write("   Estado = Índice del valor - 1\n")
            f.write("2. VALORES NEGATIVOS (<0): Nodo de Decisión Objetivo (Output).\n")
            f.write("   Acción = Índice del valor absoluto - 1\n")
            f.write("3. VALOR CERO (0): El nodo no participa en esta regla.\n\n")
            
            for name, nd in sorted(self.nodes.items()):
                if nd.kind == NodeKind.UTILITY:
                    continue
                f.write(f"Nodo: {name} ({nd.kind.name})\n")
                for i, state in enumerate(nd.states):
                    f.write(f"  {i+1} : {state}\n")
                f.write("-" * 30 + "\n")

if __name__ == "__main__":
    try:
        gen = RuleGenerator(r"example\network-bypass2.xdsl")
        gen.export_mappings(r"example\rule_mappings.txt")
        gen.generate_csv(n_rules=50, output_path=r"example\reglas_generadas.csv", respect_probs=True)
    except Exception as e:
        print(f"[!] Error: {e}")