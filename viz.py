import json
import random
from pyvis.network import Network

# Function to generate a random hex color code
def random_color():
    return "#" + ''.join(random.choices('0123456789ABCDEF', k=6))

# Create a mapping from entity type to a color
type_to_color = {}

def get_color_for_type(entity_type):
    if entity_type not in type_to_color:
        type_to_color[entity_type] = random_color()
    return type_to_color[entity_type]

# Load your data dictionary. For example, if stored in 'data.json'
file = "./data/sixth-data-extraction/related-processed-growing-tomatoes-successfully-on-the-texas-high-plains.json"
with open(file, "r", encoding="utf-8") as f:
    data = json.load(f)

# Dictionaries to hold unique nodes and list for edges
nodes = {}  # key: entity name, value: dict with type and context
edges = []  # list of tuples: (source, target, relation_detail)

# Iterate through each paragraph in your data
for paragraph in data.get("grouped_paragraphs", []):
    # Process entities
    for entity in paragraph.get("entities", []):
        name = entity.get("name")
        entity_type = entity.get("type", "Unknown")
        context = entity.get("context", "")
        if name not in nodes:
            nodes[name] = {"type": entity_type, "context": context}
    # Process relations (if available)
    for relation in paragraph.get("relations", []):
        # Each relation contains nested dictionaries for source and target entities
        source = relation.get("source_entity", {}).get("name")
        target = relation.get("target_entity", {}).get("name")
        detail = relation.get("relation_detail", "")
        if source and target:
            edges.append((source, target, detail))

# Create an interactive PyVis network
net = Network(height="750px", width="100%", notebook=True)

# Add nodes with colors based on their type
for node_name, attrs in nodes.items():
    color = get_color_for_type(attrs.get("type", "Unknown"))
    # The title attribute appears when you hover over a node
    net.add_node(node_name,
                 label=node_name,
                 title=f"Type: {attrs.get('type')}\nContext: {attrs.get('context')}",
                 color=color)

# Add edges with relation detail labels
for source, target, detail in edges:
    net.add_edge(source, target, title=detail, label=detail)

# Save and open the interactive graph in your web browser
net.show("interactive_graph.html")