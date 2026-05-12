import json
import xml.etree.ElementTree as ET
from xml.dom import minidom
import os

class Node:
    def __init__(self, name, node_type, parents, states, values):
        self.name = name
        self.type = node_type
        self.parents = parents
        self.states = states
        self.values = values

class InfluenceDiagram:
    def __init__(self):
        self.nodes = {}
        self.network_id = "modelo_exportado"

    # --- LOADERS ---
    # From json
    def load_json(self, filepath):
        self.nodes.clear()
        with open(filepath, 'r') as file:
            data = json.load(file)
            
        for k, v in data.items():
            self.nodes[k] = Node(k, v['type'], v.get('parents', []), v.get('states', []), v.get('values', []))
        print(f"[+] Modelo JSON cargado: {len(self.nodes)} nodos.")

    #From xdsl/XML
    def load_xdsl(self, filepath):
        self.nodes.clear()
        tree = ET.parse(filepath)
        root = tree.getroot()
        self.network_id = root.get('id', 'modelo_exportado')
        
        for elem in root.find('nodes'):
            node_id = elem.get('id')
            node_type = elem.tag
            
            parents_tag = elem.find('parents')
            parents = parents_tag.text.split() if parents_tag is not None else []
            
            states = [s.get('id') for s in elem.findall('state')] if node_type in ['cpt', 'decision'] else []
            
            values = []
            if node_type == 'cpt':
                values = [float(x) for x in elem.find('probabilities').text.split()]
            elif node_type == 'utility':
                values = [float(x) for x in elem.find('utilities').text.split()]
                
            self.nodes[node_id] = Node(node_id, node_type, parents, states, values)
        print(f"[+] Modelo XDSL cargado: {len(self.nodes)} nodos.")

    # --- EXPORTERS ---
    #To json
    def export_json(self, filepath):
        data = {}
        for k, n in self.nodes.items():
            data[k] = {
                'type': n.type,
                'parents': n.parents,
                'states': n.states,
                'values': n.values
            }
        with open(filepath, 'w') as file:
            json.dump(data, file, indent=4)
        print(f"[+] Exportado a JSON: {filepath}")

    #To xdsl/XML
    def export_xdsl(self, filepath):
        smile = ET.Element("smile", version="1.0", id=self.network_id, numsamples="1000", discsamples="10000")
        nodes_elem = ET.SubElement(smile, "nodes")
        
        pos_x, pos_y = 100, 100
        genie_ui = []

        for node_id, info in self.nodes.items():
            node_xml = ET.SubElement(nodes_elem, info.type, id=node_id)
            
            if info.type in ['cpt', 'decision']:
                for state in info.states:
                    ET.SubElement(node_xml, "state", id=state)

            if info.parents:
                ET.SubElement(node_xml, "parents").text = " ".join(info.parents)

            if info.type == 'cpt' and info.values:
                ET.SubElement(node_xml, "probabilities").text = " ".join(map(str, info.values))
            elif info.type == 'utility' and info.values:
                ET.SubElement(node_xml, "utilities").text = " ".join(map(str, info.values))

            # Guardar info visual para GeNIe
            genie_ui.append((node_id, info.type, pos_x, pos_y))
            pos_x += 150
            if pos_x > 800:
                pos_x = 100
                pos_y += 150

        # Bloque visual obligatorio para XDSL
        ext_elem = ET.SubElement(smile, "extensions")
        genie_elem = ET.SubElement(ext_elem, "genie", version="1.0", app="Python_Generator", name=self.network_id)
        colores = {'cpt': 'ccffff', 'decision': 'ffcc99', 'utility': 'cc99ff'}
        
        for ui_id, ui_type, x, y in genie_ui:
            ui_node = ET.SubElement(genie_elem, "node", id=ui_id)
            ET.SubElement(ui_node, "name").text = ui_id
            ET.SubElement(ui_node, "interior", color=colores.get(ui_type, 'ffffff'))
            ET.SubElement(ui_node, "outline", color="000080")
            ET.SubElement(ui_node, "font", color="000000", name="Arial", size="14")
            ET.SubElement(ui_node, "position").text = f"{x} {y} {x+100} {y+60}"

        ET.indent(smile, space="\t", level=0)
        xml_str = ET.tostring(smile, encoding='unicode', xml_declaration=True)
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(xml_str)
        print(f"[+] Exportado a XDSL: {filepath}")


# 1. Instanciar la red
mi_modelo = InfluenceDiagram()

# 2. Cargar (elige la que prefieras)
mi_modelo.load_xdsl('example/network-bypass2.xdsl')
# mi_modelo.load_json('red.json')

# 3. Exportar a otros formatos si lo necesitas
mi_modelo.export_json('copia_modelo.json')
mi_modelo.export_xdsl('copia_modelo.xdsl')
