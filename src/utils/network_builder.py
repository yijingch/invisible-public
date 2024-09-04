import numpy as np
import pandas as pd
from itertools import combinations
from typing import Any, List, Dict, Union
import networkx as nx

from src.utils.functions import get_cosine_similarity, get_jaccard_similarity, normalize


def convert_list_to_vector(item_ls, item2idx):
    vec = np.zeros(len(item2idx))
    for i in item_ls:
        vec[item2idx[i]] += 1
    return vec


def build_projection(
        df:pd.DataFrame,
        node_col:str,
        feature_col:str,
        similarity_func:Any = get_cosine_similarity, # TODO: validate for jaccard similarity too
        density_thres:float = -1,
        node_label_map:Dict = {},
        node_labels:List = [],
        shuffle_mode:str = "",
        save_fpath:str = "",
        print_info:bool = True,
        save_num_idx:bool = False) -> Union[pd.DataFrame, pd.DataFrame]:
    """Create a projection network of a given aggregation unit (usually user).

    Args:
        df (pd.DataFrame): the input dataframe based on which to build the network
        col_to_aggr (str): the aggregation unit (usually user)
        col_as_feature (str): the feature for each aggregation unit (the activity)
        similarity_func (Any, optional): the similarity function to use. Defaults to get_cosine_similarity.
        density_thres (float, optional): the density threshold to extract only meaningful links. Defaults to -1.
        node_label_map (Dict, optional): the node label dictionary to set attributes. Defaults to {}.
        node_labels (List, optional): the node labels to add. Defaults to [].
        shuffle_mode (str, optional): the mode of shuffling, if set with a string then we'll produce a shuffled network. Defaults to "".
        save_fpath (str, optional): the output path to save. Defaults to "".
        print_info (bool, optional): whether to print network description. Defaults to True.

    Returns:
        [pd.DataFrame, pd.DataFrame]: the node_df and edge_df for the network
    """

    feature_unit = df[feature_col].unique()
    feat2idx = {}
    idx2feat = {}
    for i,f in enumerate(feature_unit):
        feat2idx[f] = i 
        idx2feat[i] = f 

    if shuffle_mode == "before_aggr":
        df[node_col] = np.random.permutation(df[node_col])
    df_aggr = df.groupby(node_col).agg({feature_col: lambda x: list(x)}).reset_index()
    if shuffle_mode == "after_aggr":
        df_aggr[node_col] = np.random.permutation(df_aggr[node_col])

    df_aggr["feature_vec"] = df_aggr[feature_col].map(lambda x: convert_list_to_vector(x, feat2idx))
    df_aggr["feature_vec_norm"] = df_aggr["feature_vec"].map(lambda x: normalize(x))
    df_aggr["feature_sum"] = df_aggr["feature_vec"].map(lambda x: np.sum(x))

    node_df = df_aggr[[node_col, "feature_sum"]].rename(columns={node_col:"Id"})
    node_df["Label"] = node_df["Id"]

    sources = []
    targets = []
    weights = []
    nodes = node_df["Id"].tolist()
    feature_map = df_aggr.set_index(node_col).to_dict()["feature_vec_norm"]
    for n1, n2 in combinations(nodes, 2):
        vec1 = feature_map[n1]
        vec2 = feature_map[n2]
        sources.append(n1)
        targets.append(n2)
        weights.append(similarity_func(vec1, vec2))
    edge_df = pd.DataFrame()
    edge_df["Source"] = sources 
    edge_df["Target"] = targets
    edge_df["weight"] = weights 

    if density_thres > 0:
        n_nodes = len(node_df)
        n_edges = round(n_nodes*(n_nodes-1)*density_thres)
        thres = sorted(weights, reverse=True)[n_edges]
        if thres > 0:
            edge_df = edge_df[edge_df["weight"]>=thres]
        else:
            edge_df = edge_df[edge_df["weight"]>0]
        
    if print_info:
            print("# of nodes:", len(node_df))
            print("# of edges:", len(edge_df))

    if len(node_labels) > 0:
        for lab in node_labels:
            node_df[lab] = node_df["Id"].map(lambda x: node_label_map[x][lab])

    if save_num_idx:
        node_df = node_df.reset_index()
        node_df["Id"] = node_df["index"]
        edge_df = edge_df.merge(node_df[["index", "Label"]], left_on="Target", right_on="Label", how="left")
        edge_df = edge_df.drop(columns="Target").rename(columns={"index":"Target", "Label":"Label_target"})
        edge_df = edge_df.merge(node_df[["index", "Label"]], left_on="Source", right_on="Label", how="left")
        edge_df = edge_df.drop(columns="Source").rename(columns={"index":"Source", "Label":"Label_source"})

    if len(save_fpath) > 0:
        node_df.to_csv(save_fpath + f"projnet_{node_col}_{feature_col}_density{density_thres}_NODE.csv", index=False)
        edge_df.to_csv(save_fpath + f"projnet_{node_col}_{feature_col}_density{density_thres}_EDGE.csv", index=False)

    return node_df, edge_df


def read_network(
        node_df:pd.DataFrame, 
        edge_df:pd.DataFrame, 
        add_node_attrs:bool = True,
        add_edge_attrs:bool = False,
        print_info:bool = True,
        n_sample_nodes:int = -1,
        ) -> nx.Graph:
    """Create an nx.Graph object for a given set of nodes and edges

    Args:
        node_df (pd.DataFrame): a table listing out all nodes, columns should contain `Id`, `Label`, `attr1` (optional), `attr2` (optional)...
        edge_df (pd.DataFrame): a table listing out all edges, columns should contain `Source`, `Target`, `attr1` (optional), `attr2` (optional)...
        add_node_attrs (bool, optional): whether to write node attributes to the graph. Defaults to True.
        add_edge_attrs (bool, optional): whether to write edge attributes to the graph. Defaults to False.
        print_info (bool, optional): whether to print network description. Defaults to True.

    Returns:
        nx.Graph: a x.Graph object for the output network
    """
    node_attr_cols = []
    edge_attr_cols = []

    if n_sample_nodes > 0:
        node_df = node_df.sample(n=n_sample_nodes, replace=False)
        nodes = node_df["Id"].tolist()
        edge_df = edge_df[(edge_df["Source"].isin(nodes))&(edge_df["Target"].isin(nodes))]
            
    if add_node_attrs:
        node_attr_cols = set(node_df.columns) - {"Id", "Label"}
        if len(node_attr_cols) == 0:
            print("Node attributes not found!")
    if add_edge_attrs:
        edge_attr_cols = set(edge_df.columns) - {"Source", "Target"}
        if len(node_attr_cols) == 0:
            print("Edge attributes not found!")
    node_list = []
    edge_list = []
    for _,row in node_df.iterrows():
        if len(node_attr_cols) > 0:
            node_list.append((row["Id"], {c:row[c] for c in node_attr_cols}))
        else:
            node_list.append(row["Id"])
    for _,row in edge_df.iterrows():
        if len(edge_attr_cols) > 0:
            edge_list.append((row["Source"], row["Target"], {c:row[c] for c in edge_attr_cols}))
        else:
            edge_list.append((row["Source"], row["Target"]))
        
    g = nx.Graph()
    g.add_nodes_from(node_list)
    g.add_edges_from(edge_list)
    
    if print_info:
        print(nx.info(g))

    # sanity check!
    assert len(node_df) == len(g.nodes), f"Node exporting error; have {len(node_df)} rows in the input but have {len(g.nodes)} nodes in the network."
    assert len(edge_df) == len(g.edges), f"Edge exporting error; have {len(edge_df)} rows in the input but have {len(g.edges)} edges in the network."

    return g