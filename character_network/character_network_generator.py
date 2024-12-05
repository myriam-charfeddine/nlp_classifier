import pandas as pd
import networkx as nx
from pyvis.network import Network

class CharacterNetworkGenerator:
    def __init__(self) :
        pass

    def character_network_generator(self, df):
        window = 10 #nb of sentences gap range (from 1 to 10) that an entity is considered related to another
        entity_relationship = []

        for row in df['Ners']:
            previous_entities_in_window= []

            for sentence in row:
                previous_entities_in_window.append(list(sentence))
                previous_entities_in_window = previous_entities_in_window[-window :] #only the last 10 sentences are considered for the current one(the 10 closest to the current one)
                
                #Flatten 2D list into 1D list:
                previous_entities_in_window_flattened = sum(previous_entities_in_window, []) #input for SUM should be lists

                for entity in sentence:
                    for entity_in_window in previous_entities_in_window_flattened:
                        if entity != entity_in_window:
                            entity_relationship.append(sorted([entity, entity_in_window])) #sorted so [A,B] and [B,A] are considered the same

        
        relationship_df = pd.DataFrame({'Value': entity_relationship})
        relationship_df['Source'] = relationship_df['Value'].apply(lambda x: x[0])
        relationship_df['Target'] = relationship_df['Value'].apply(lambda x: x[1])

        relationship_df = relationship_df.groupby(['Source', 'Target']).count().reset_index()
        relationship_df = relationship_df.sort_values('Value', ascending=False)

        return relationship_df
    
    def draw_network_graph(self, relationship_df):
        relationship_df = relationship_df.sort_values('Value', ascending=False)
        relationship_df = relationship_df.head(200) #limit it to only the top 200 pairs so the graph could be clearer

        Graph = nx.from_pandas_edgelist(
        relationship_df.head(200),
        source='Source',
        target='Target',
        edge_attr='Value',
        create_using=nx.Graph()
        )

        net = Network(notebook=True, width="1000px", height='700px', bgcolor="#222222", font_color="white", cdn_resources="remote")
        node_degree = dict(Graph.degree)

        nx.set_node_attributes(Graph, node_degree, 'size')
        net.from_nx(Graph)
        # net.show("Naruto.html")

        html = net.generate_html()
        html = html.replace("'","\"") #to avoid any format conflict when using iframe afterward

        output_html = f"""<iframe style="width: 100%; height: 600px;margin:0 auto" name="result" allow="midi; geolocation; microphone; camera;
        display-capture; encrypted-media;" sandbox="allow-modals allow-forms
        allow-scripts allow-same-origin allow-popups
        allow-top-navigation-by-user-activation allow-downloads" allowfullscreen=""
        allowpaymentrequest="" frameborder="0" srcdoc='{html}'></iframe>"""

        #Note for myself: An iframe (inline frame) is an HTML element used to embed another HTML document or web page within the current page. 
        
        return output_html