import dash
import dash_cytoscape as cyto
from dash import html, dcc
from dash.dependencies import Input, Output, State
from linixicon import WordEmbedding
import logging
import networkx as nx
from collections import deque

# Set up logging
logging.basicConfig(level=logging.DEBUG)

embedding_model = WordEmbedding()
threshold_percent = 50

def get_two_random_words_below_threshold(model, threshold):
    while True:
        word1, word2 = model.select_random_words(2)
        similarity = model.find_similarity_cosine(word1, word2)
        if similarity < threshold:
            return word1, word2

def bfs(graph, start, end):
    queue = deque([[start]])
    visited = set([start])
    
    while queue:
        path = queue.popleft()
        node = path[-1]
        
        if node == end:
            return path
        
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                new_path = list(path)
                new_path.append(neighbor)
                queue.append(new_path)
    
    return None

def check_win_condition(elements, start_node, end_node):
    graph = nx.Graph()
    for element in elements:
        if 'source' in element['data'] and 'target' in element['data']:
            graph.add_edge(element['data']['source'], element['data']['target'])
        elif 'id' in element['data']:
            graph.add_node(element['data']['id'])
    
    path = bfs(graph, start_node, end_node)
    return path is not None

app = dash.Dash(__name__)

app.layout = html.Div([
    html.Button('Add Two Random Nodes', id='add-nodes-button'),
    dcc.Input(id='node-name-input', type='text', placeholder='Enter node name'),
    html.Button('Add Node', id='add-node-button'),
    html.Div(id='suggestions', style={'color': 'red'}),
    html.Div(id='win-message', style={'color': 'green', 'font-weight': 'bold'}),
    cyto.Cytoscape(id='cytoscape', elements=[], layout={'name': 'circle'}),
    dcc.Store(id='initial-nodes')
])

@app.callback(
    Output('cytoscape', 'elements'),
    Output('initial-nodes', 'data'),
    Output('win-message', 'children'),
    Input('add-nodes-button', 'n_clicks'),
)
def add_two_random_nodes(n_clicks):
    logging.debug('Adding two random nodes...')
    
    word1, word2 = get_two_random_words_below_threshold(embedding_model, threshold_percent)
    logging.debug(f'Selected words: {word1}, {word2}')
    
    new_elements = [
        {'data': {'id': word1, 'label': word1}},
        {'data': {'id': word2, 'label': word2}}
    ]
    return new_elements, {'start': word1, 'end': word2}, ""

@app.callback(
    Output('cytoscape', 'elements', allow_duplicate=True),
    Output('suggestions', 'children'),
    Output('win-message', 'children', allow_duplicate=True),
    Input('add-node-button', 'n_clicks'),
    State('node-name-input', 'value'),
    State('cytoscape', 'elements'),
    State('initial-nodes', 'data'),
    prevent_initial_call=True
)
def add_node(n_clicks, node_name, elements, initial_nodes):
    if n_clicks is None or node_name is None or node_name.strip() == "":
        raise dash.exceptions.PreventUpdate

    node_name = node_name.strip()
    logging.debug(f'Attempting to add node: {node_name}')
    
    try:
        if node_name in embedding_model.token2int:
            logging.debug(f'Word {node_name} found in embedding model.')
            new_node = {'data': {'id': node_name, 'label': node_name}}
            elements.append(new_node)
            
            # Check similarity with existing nodes and create edges if necessary
            highest_similarity = 0
            for element in elements:
                if 'source' not in element['data'] and 'target' not in element['data']:
                    existing_node = element['data']['id']
                    if existing_node != node_name:  # Prevent self-connection
                        similarity = embedding_model.find_similarity_cosine(node_name, existing_node)
                        if similarity > threshold_percent:
                            logging.debug(f'Creating edge between {node_name} and {existing_node} with similarity {similarity}')
                            elements.append({'data': {'source': node_name, 'target': existing_node}})
                            highest_similarity = max(highest_similarity, similarity)
            
            logging.debug(f'Highest similarity for {node_name} was {highest_similarity}')
            
            # Check win condition
            if check_win_condition(elements, initial_nodes['start'], initial_nodes['end']):
                win_message = "Congratulations! You've connected the initial nodes and won the game!"
            else:
                win_message = ""
            
            return elements, "", win_message
        else:
            logging.debug(f'Word {node_name} not found. Suggesting alternatives.')
            suggestions = embedding_model.suggest_words(node_name, list(embedding_model.token2int.keys()))
            suggestion_text = "Word not found. Did you mean: " + ", ".join(suggestions)
            return elements, suggestion_text, ""
    except Exception as e:
        logging.error(f"Error adding node: {e}")
        return elements, f"Error adding node: {e}", ""

if __name__ == '__main__':
    app.run_server(debug=True)