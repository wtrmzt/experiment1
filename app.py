import dash
from dash import dcc, html, Input, Output, State
import visdcc
import os
from datetime import datetime
from dash.exceptions import PreventUpdate
import json
from openai import OpenAI
import re
import ast
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

APIKEY=os.getenv("OPENAI_API_KEY")

app = dash.Dash(__name__)



# CSVデータの読み込み
def relate_map(nodes, edges):
    #new_nodes = [{'id': node['id'], 'label': f"Modified {node['label']}"} for node in nodes]
    node_name=nodes[0]['label']
    subject_name=nodes[0]['id']
    print('node'+node_name)
    print('subject'+subject_name)
    new_map = text2dic(relate_GPToutput(node_name),subject_name)
    new_map = rename_id_added(new_map,subject_name)
    #node_name=nodes['label']
    return new_map

def reflection_map(text):
    subject_name='振り返り'
    new_map = text2dic(reflection_GPToutput(text),subject_name)
    new_map = rename_id(new_map,subject_name)
    return new_map

def relate_GPToutput(input_name):
    client = OpenAI(api_key=APIKEY)
    node_gpt_output=[]
    res = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": '''#命令
    あなたは優秀な教員です。以下の条件に従い、最善の出力をしてください。'''},  # 役割設定
                {"role": "user", "content": '''
    #条件
    入力として、単元名が1つ与えられる。これを基に関連項目を示すように知識マップを作成する。知識マップ作成の条件は、以下である。
    ・中心のノードは単元名。
    ・高さ1の知識マップを作成。
    ・それぞれのノードに対して、記述を基にして140字以内で説明文を生成せよ。
    ・学習するにあたってその単元を深める項目であること

    #出力
    PythonのNetworkXライブラリで読み込み可能な、nodes辞書と、edges辞書の2つ。
    ・nodes=[{'id':i,'label':"node_name",'sentence':"writetext"}]：ノードが格納される。idにはノードの番号を格納。labelにはノード名、sentenceには説明文を140字以内で格納する。

    ・edges=[{'from':node_id,'to':node_id}]：fromにはエッジの始点のノードid、toにはエッジの終点のノードidを格納する。
    ・これ以外は必要ない。

    #入力
                '''+input_name}               # 最初の質問
            ],
            temperature=0.0  # 温度（0-2, デフォルト1）
        )
    node_gpt_output.append(res.choices[0].message.content)
    return node_gpt_output

def clean_list_comments(input_str):
    # 正規表現パターン定義
    pattern = re.compile(r'\[.*?\]', re.DOTALL)
    
    # リスト部分を全て取得
    lists = pattern.findall(input_str)
    
    cleaned_lists = []
    
    for lst in lists:
        # # から始まる行を削除
        lst_cleaned_comments = re.sub(r'#.*?\n', '', lst)
        
        # ] の前にある , を削除
        lst_cleaned_comma = re.sub(r',\s*]', ']', lst_cleaned_comments)
        
        cleaned_lists.append(lst_cleaned_comma)
    
    # 元の文字列にクリーンなリストを置き換える
    for original, cleaned in zip(lists, cleaned_lists):
        input_str = input_str.replace(original, cleaned)
    
    return input_str
    
def text2dic(node_gpt_output,subject_name):
    node_gpt_map=[]
    for i in range(len(node_gpt_output)):
        # 正規表現でノードとエッジの部分を抽出
        nodes_pattern = re.compile(r'nodes = \s*(\[\s*\{.*?\}\s*\])', re.DOTALL)
        edges_pattern = re.compile(r'edges = \s*(\[\s*\{.*?\}\s*\])', re.DOTALL)
        #print('nodes'+node_gpt_output[i])
        nodes_match = nodes_pattern.search(clean_list_comments(node_gpt_output[i]))
        edges_match = edges_pattern.search(clean_list_comments(node_gpt_output[i]))

        if nodes_match:
            nodes_str = nodes_match.group(1)
            nodes = ast.literal_eval(nodes_str)
        else:
            #print(i)
            print("Nodes情報が見つかりませんでした。")

        if edges_match:
            edges_str = edges_match.group(1)
            edges = ast.literal_eval(edges_str)
        else:
            #print(i)
            print("Edges情報が見つかりませんでした。")
            #print(nodes)
        node_gpt_map.append({'nodes':nodes,'edges':edges,'subject':subject_name})
    #print(node_gpt_map)
    #node_gpt_map=rename_id_added(node_gpt_map,subject_name)
    #print(node_gpt_map)

    return node_gpt_map
def rename_id_added(map,name):
    color = "#FFE568"
    map=map[0]
    root = map['nodes'][0]['id']
    map['nodes'][0]['id'] = name# +'_'+str(map['nodes'][0]['id'])
    map['nodes'][0]['color']=color
    print(map)
    for i in range(1,len(map['nodes'])):
        map['nodes'][i]['id'] = name + '_' + str(map['nodes'][i]['id']) 
        map['nodes'][i]['color'] = color

    for j in range(len(map['edges'])):
        if map['edges'][j]['from'] == root:
            map['edges'][j]['from'] = name# + '_' + str(map['edges'][j]['from'])
            map['edges'][j]['to'] = name + '_' + str(map['edges'][j]['to'])
        elif map['edges'][j]['to'] == root:
            map['edges'][j]['from'] = name + '_' + str(map['edges'][j]['from'])
            map['edges'][j]['to'] = name# + '_' + str(map['edges'][j]['to'])
        else:
            map['edges'][j]['from'] = name + '_' + str(map['edges'][j]['from'])
            map['edges'][j]['to'] = name + '_' + str(map['edges'][j]['to'])
    return map

def rename_id(map,name):
    print(map)
    map=map[0]
    color = {'情報I':'#A0D8EF','コンピューターリテラシー':'#F8C6BD','プログラミング通論':'#E3EBA4','美術A':'#FFFFFF','振り返り':'#FFFFFF'}
    for i in range(len(map['nodes'])):
        map['nodes'][i]['id'] = name + '_' + str(map['nodes'][i]['id'])
        map['nodes'][i]['color'] = color[name]
    for j in range(len(map['edges'])):
        map['edges'][j]['from'] = name + '_' + str(map['edges'][j]['from'])
        map['edges'][j]['to'] = name + '_' + str(map['edges'][j]['to'])
    return map

def reflection_GPToutput(input_text):
    client = OpenAI(api_key=APIKEY)
    node_gpt_output=[]
    res = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": '''#命令
    あなたは優秀な教員です。以下の条件に従い、最善の出力をしてください。'''},  # 役割設定
                {"role": "user", "content": '''
#条件
入力に生徒の振り返り記述の内容が書かれる。この記述を基に、"分野ごと"に木を伸ばす知識マップを作成しなさい。
・それぞれのノードに対して、140字以内で説明文を生成せよ。
・ルートノードは、振り返り単元である。
・「課題」・「演習」などのノードは使用しない
・ノード名は簡潔にせよ
・高さは自由である．必要に応じて伸ばして良い
・作成した木を見返してもいいように、'学習の理解を深める体系的な内容のみ'であること。
    #出力
    PythonのNetworkXライブラリで読み込み可能な、nodes辞書と、edges辞書の2つ。
    ・nodes=[{'id':i,'label':"node_name",'sentence':"writetext"}]：ノードが格納される。idにはノードの番号を格納。labelにはノード名、sentenceには説明文を140字以内で格納する。

    ・edges=[{'from':node_id,'to':node_id}]：fromにはエッジの始点のノードid、toにはエッジの終点のノードidを格納する。
    ・これ以外は必要ない。

    #入力
                '''+input_text}               # 最初の質問
            ],
            temperature=0.0  # 温度（0-2, デフォルト1）
        )
    node_gpt_output.append(res.choices[0].message.content)
    return node_gpt_output

def save_user_data(name_input,reflection_text,data):
    folder_name = f"{name_input}_ex"
    os.makedirs(folder_name, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 振り返り文章を保存
    with open(os.path.join(folder_name, f"reflection_{timestamp}.txt"), "w", encoding="utf-8") as f:
        f.write(reflection_text or "")

    # マップデータを保存
    with open(os.path.join(folder_name, f"map_{timestamp}.json"), "w", encoding="utf-8") as f:
        import json
        json.dump(data, f, ensure_ascii=False, indent=2)

"""
関連の算出
"""
def get_embedding_small(text, model="text-embedding-3-small"):
    client = OpenAI(api_key=APIKEY)

    text = text.replace("\n", " ")
    return client.embeddings.create(input = [text], model=model).data[0].embedding

# 学年でノードをフィルタリングする関数（改良済み）
def filter_nodes_by_year(nodes_df, user_year):
    def extract_min_year(years):
        if isinstance(years, str) and '/' in years:
            try:
                return min(map(int, years.split('/')))
            except ValueError:
                return float('inf')
        try:
            return int(years)
        except ValueError:
            return float('inf')

    def is_valid_year(years, user_year):
        min_year = extract_min_year(years)
        return min_year <= int(user_year)

    return nodes_df[nodes_df['year'].apply(lambda y: is_valid_year(y, user_year))]

# コサイン類似度で最も近いノードを選ぶ関数
def find_similar_nodes(embedding, nodes_df, top_n=5):
    """
    振り返り文のベクトルとすべてのノードのベクトルから、類似度上位n個を選出
    """
    node_embeddings = np.vstack(nodes_df['embedding'].to_list())
    #print([ast.literal_eval(sublist[0]) for sublist in node_embeddings])
    similarities = cosine_similarity([embedding],  [ast.literal_eval(sublist[0]) for sublist in node_embeddings]).flatten()
    top_indices = similarities.argsort()[-top_n:][::-1]
    return nodes_df.iloc[top_indices]

# CSVデータの読み込み
def load_csv_data(subjectname):
    nodes_file = './UECsubject_maps3/subject_map_'+ subjectname +'_nodes.csv'  # CSVファイルのパス
    edges_file = './UECsubject_maps3/subject_map_'+subjectname+'_edges.csv'
    nodes = []
    edges = []
    if subjectname=='リセット':
        return nodes,edges
    nodes_df = pd.read_csv(nodes_file)
    for index, row in nodes_df.iterrows():
        nodes.append({'id': row['id'], 'label': row['label'], 'color': row['color']})

    edges_df = pd.read_csv(edges_file)
    for index, row in edges_df.iterrows():
        edges.append({'from': row['from'], 'to': row['to']})

    return nodes, edges

# エッジを作成する関数
def create_edges(reflection_node, similar_nodes):
    """
    振り返りノードと類似ノードをつなぐエッジを作成
    """
    nodes=[]
    edges = []
    reflection_id = reflection_node['id']
    for _, row in similar_nodes.iterrows():
        edges.append({'from': reflection_id, 'to': row['id']})
        match = re.match(r"(.+)_\d+", row['id'])
        tmp_node, tmp_edge = load_csv_data(match.group(1))
        edges.extend(tmp_edge)
        nodes.extend(tmp_node)
    return nodes,edges

def get_related_nodes(subject_nodes_df, all_nodes_df):
    """
    subject_nodes_dfの各要素を基に関連するall_nodes_dfの行を取得し、データフレーム形式で出力

    Parameters:
    - subject_nodes_df: 科目ノードのデータフレーム
    - all_nodes_df: すべてのノードを含むデータフレーム
    - csv_to_nodes: ファイル名とそのノードデータを格納した辞書

    Returns:
    - related_data_df: subject_idごとに関連するall_nodes_dfの行を含むデータフレーム
    """
    related_rows = []

    for _, row in subject_nodes_df.iterrows():
        subject_id = row['id']  # 科目IDを取得
        #print(row)
        # csv_to_nodesを検索し、subject_idが含まれるファイルを特定
        file_name = row['source_file']
        # 該当する行をall_nodes_dfから取得し、追加
        file_related_rows = all_nodes_df[all_nodes_df['source_file'] == file_name]
        #file_related_rows = file_related_rows[file_related_rows['id'] == subject_id]
        related_rows.append(file_related_rows)

    # すべての関連行を結合し、データフレーム形式で出力
    related_data_df = pd.concat(related_rows, ignore_index=True)
    return related_data_df

# マップを生成して保存する関数
def generate_maps_from_reflections(reflection_embeddings, subject_name, syllabus_dir, output_dir, user_year, top_m, top_n):
    """
    振り返り文の各ノードと全科目のノードを接続し、新しいマップを生成
    """
     # 出力フォルダを作成
    os.makedirs(output_dir, exist_ok=True)

    # すべてのノードとエッジを収集
    all_nodes = []
    subject_nodes = []
    csv_to_nodes = {}  # ファイル名とそのノードデータを対応付ける辞書

    for file in os.listdir(syllabus_dir):
        if file.endswith("_nodes.csv"):
            nodes_df = pd.read_csv(os.path.join(syllabus_dir, file))

            # 科目埋め込み (2行目) を抽出して保存
            subject_row = nodes_df.iloc[[0]]  # 2行目（index 1）を取得
            # ファイル名をsource_file列に追加
            subject_row['source_file'] = file

            subject_nodes.append(subject_row)
            # 年度でフィルタリングしたノードを収集
            filtered_nodes = filter_nodes_by_year(nodes_df, user_year)

            # ファイル名をsource_file列として追加
            filtered_nodes['source_file'] = file
            all_nodes.append(filtered_nodes)

            # ファイル名とそのノードを辞書に格納
            csv_to_nodes[file] = filtered_nodes

    # データフレームに結合
    all_nodes_df = pd.concat(all_nodes, ignore_index=True)
    subject_nodes_df = pd.concat(subject_nodes, ignore_index=True)

    # 振り返り文ごとに処理
    all_new_edges = []
    all_new_nodes = []

    subject_embedding = get_embedding_small(subject_name)

    for idx, reflection_embedding in enumerate(reflection_embeddings):
        # 振り返り文をベクトル化
        embedding = get_embedding_small(reflection_embedding['label'])
        reflection_node = {
            'id': reflection_embedding['id'],
            'label': reflection_embedding['label'],
            'sentence': '実験',
            'color': 'yellow',
            'embedding': embedding,
            'year': 'Reflection'
        }

        # 類似ノードを選出（科目ごと）
        #similar_nodes = find_similar_nodes(embedding, all_nodes_df, top_n)
        similar_nodes_subject = find_similar_nodes(subject_embedding, subject_nodes_df, top_m)
        similar_nodes = find_similar_nodes(embedding,get_related_nodes(similar_nodes_subject,all_nodes_df),top_n)

        # 振り返りノードと類似ノードを接続
        add_node,new_edges = create_edges(reflection_node, similar_nodes)

        # 振り返りノードを追加
        all_new_nodes.append(reflection_node)
        all_new_nodes.extend(similar_nodes.to_dict('records'))
        all_new_edges.extend(new_edges)
        all_new_nodes.extend(add_node)
        print('add_nodes blow')
        #display(add_node)

    #display(all_new_nodes)
    print('Frag')

        # 新しいノードとエッジを保存
    #all_new_nodes_df = pd.DataFrame(all_new_nodes)
    #all_new_edges_df = pd.concat(all_new_edges, ignore_index=True)

    #display(all_new_nodes_df)
    print('generate complete')
    return {'nodes':all_new_nodes,'edges':all_new_edges}

# ダミーデータ（ノードとエッジ）を初期設定
initial_nodes = [
    {'id': '1', 'label': '条件分岐', 'sentence': 'This is the sentence for Node 1.'},
    {'id': '2', 'label': 'Node 2', 'sentence': 'This is the sentence for Node 2.'}
]
initial_edges = [{'from': '1', 'to': '2'}]

# クラシックなデザインスタイルを適用
app.layout = html.Div([
    # サイドバー
    html.Div(style={'width': '5%', 'height': '100vh', 'backgroundColor': '#7D7461', 'float': 'left', 'padding': '0'}, 
             children=[
                 html.Div(style={'padding': '10px', 'textAlign': 'center', 'fontSize': '24px', 'fontFamily': 'Garamond', 'color': '#FFF'}, children=[
                     html.Span("≡")
                 ])
             ]),
    
    # メインコンテンツ
    html.Div([
        # マップの表示部分
        html.Div([
            html.H3('マップの表示部分', style={'padding': '10px', 'fontFamily': 'Garamond', 'color': '#7D7461', 'margin': '0'}),
            visdcc.Network(
                id='net',
                data={'nodes': initial_nodes, 'edges': initial_edges},
                options={'height': '600px', 'width': '100%', 'clickToUse': True, 'physics': {'barnesHut': {'avoidOverlap': 0}}},
                selection={'nodes': [], 'edges': []}
            ),
            html.Div([
                
                #html.Button('+', id='zoom-in', style={'fontSize': '24px', 'margin': '5px', 'backgroundColor': '#D2B48C', 'color': '#7D7461', 'border': 'none', 'borderRadius': '5px', 'padding': '5px 15px'}),
                #html.Button('-', id='zoom-out', style={'fontSize': '24px', 'margin': '5px', 'backgroundColor': '#D2B48C', 'color': '#7D7461', 'border': 'none', 'borderRadius': '5px', 'padding': '5px 15px'})
                
            ], style={'textAlign': 'center', 'padding': '10px'})
        ], style={'width': '100%', 'height': '60vh', 'float': 'left', 'border': '2px dashed #7D7461', 'backgroundColor': '#F4EBD9', 'padding': '0', 'position': 'relative'}),
        
        # ノードの表示・説明部分（ポップアップ）
        html.Div([
            html.Div([
                html.Button('×', id='close-button', style={'float': 'right', 'fontSize': '18px', 'backgroundColor': '#7D7461', 'color': '#FFF', 'border': 'none', 'borderRadius': '5px', 'padding': '5px 10px', 'fontFamily': 'Garamond'}),
                html.H3('ノードの表示・説明', style={'backgroundColor': '#B3A79A', 'padding': '10px', 'color': '#FFF', 'textAlign': 'center', 'fontFamily': 'Garamond', 'margin': '0'}),
                html.Div(id='node-info', style={'padding': '10px', 'backgroundColor': '#F0D9A9', 'height': 'auto', 'fontFamily': 'Garamond'}),
                html.Div([
                    html.Button('周辺情報取得', id='get-info-button', style={'fontSize': '16px', 'margin': '5px', 'backgroundColor': '#D2B48C', 'color': '#7D7461', 'border': 'none', 'borderRadius': '5px', 'padding': '5px 10px'}),
                    html.Button('反映', id='reflect-button', style={'fontSize': '16px', 'margin': '5px', 'backgroundColor': '#D2B48C', 'color': '#7D7461', 'border': 'none', 'borderRadius': '5px', 'padding': '5px 10px'})
                ], style={'padding': '10px', 'textAlign': 'center'}),
            ], style={'height': 'auto'})
        ], id='node-popup', style={
            'width': '20%', 'height': 'auto', 'position': 'absolute', 'top': '10%', 
            'right': '-30%', 'border': '2px dashed #7D7461', 'padding': '0', 
            'backgroundColor': '#F4EBD9', 'transition': 'right 0.5s cubic-bezier(0.25, 1, 0.5, 1)', 
            'display': 'none'  # 初期状態では非表示
        }),

        # ボタンの配置をメインコンテンツ部分に追加
        html.Div([
            html.Div(id='loading-message', style={'fontFamily': 'Garamond', 'color': '#7D7461', 'fontSize': '18px', 'textAlign': 'center', 'margin': '10px'}),
        ], style={'textAlign': 'center', 'padding': '10px'}),
        # 振り返りの文章入力
        html.Div([
            html.H3('振り返りの文章入力', style={'padding': '10px', 'fontFamily': 'Garamond', 'color': '#7D7461', 'margin': '0'}),
            html.Label('名前を記入してください', style={'fontFamily': 'Garamond', 'color': '#7D7461'}),
            dcc.Input(id='name-input', type='text', placeholder='名前を入力...', style={'width': '30%', 'fontFamily': 'Garamond', 'padding': '10px', 'border': '2px solid #7D7461', 'backgroundColor': '#F4EBD9'}),
            html.Label('学年を記入してください', style={'fontFamily': 'Garamond', 'color': '#7D7461'}),
            dcc.Input(id='year-input', type='text', placeholder='学年を入力...', style={'width': '20%', 'fontFamily': 'Garamond', 'padding': '10px', 'border': '2px solid #7D7461', 'backgroundColor': '#F4EBD9'}),
            dcc.Textarea(
                id='reflection-text',
                placeholder='ここにテキストを入力してください...',
                style={'width': '90%', 'height': '100px', 'fontFamily': 'Garamond', 'border': '2px solid #7D7461', 'padding': '10px', 'backgroundColor': '#F4EBD9'}
            ),
            html.Div([
                html.Button('生成', id='generate-graph-button', style={'fontSize': '16px', 'margin': '10px', 'backgroundColor': '#D2B48C', 'color': '#7D7461', 'border': 'none', 'borderRadius': '5px', 'padding': '5px 15px'}),
                html.Button('関連の取得', id='relate-graph-button', style={'fontSize': '16px', 'margin': '10px', 'backgroundColor': '#D2B48C', 'color': '#7D7461', 'border': 'none', 'borderRadius': '5px', 'padding': '5px 15px'}),
                html.Button('保存', id='save-button', style={'fontSize': '16px', 'margin': '10px', 'backgroundColor': '#D2B48C', 'color': '#7D7461', 'border': 'none', 'borderRadius': '5px', 'padding': '5px 15px'})
            ], style={'padding': '10px', 'textAlign': 'center'})
        ], style={'width': '100%', 'height': 'auto', 'clear': 'both', 'border': '2px dashed #7D7461', 'backgroundColor': '#F4EBD9', 'padding': '0'})
    ], style={'width': '95%', 'float': 'right', 'padding': '0'}),
])
@app.callback(
    Output('net', 'data', allow_duplicate=True),
    Output('loading-message', 'children'),
    Input('generate-graph-button', 'n_clicks'),
    Input('relate-graph-button', 'n_clicks'),
    Input('get-info-button', 'n_clicks'),
    State('reflection-text', 'value'),
    State('net', 'selection'),
    State('name-input', 'value'),
    State('year-input', 'value'),
    State('net', 'data'),
    prevent_initial_call=True
)
def handle_buttons(generate_clicks,relate_clicks,get_info_click, input_text, selection, name_input, user_year, net_data):
    ctx = dash.callback_context

    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate

    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    loading_message = ''
    if button_id == 'generate-graph-button':
        if not input_text:
            raise dash.exceptions.PreventUpdate
        
        # ロードメッセージを設定
        loading_message = '追加中...'
        
        # reflection_map関数で新しいノードとエッジを生成
        new_map = reflection_map(input_text)
        
        # ノードの更新
        new_nodes = {node['id']: node for node in new_map['nodes']}
        for i, node in enumerate(net_data['nodes']):
            if node['id'] in new_nodes:
                net_data['nodes'][i] = new_nodes.pop(node['id'])
        net_data['nodes'].extend(new_nodes.values())
        
        # エッジの更新
        new_edges = {(edge['from'], edge['to']): edge for edge in new_map['edges']}
        for i, edge in enumerate(net_data['edges']):
            edge_tuple = (edge['from'], edge['to'])
            if edge_tuple in new_edges:
                net_data['edges'][i] = new_edges.pop(edge_tuple)
        net_data['edges'].extend(new_edges.values())
        
        loading_message = ''
        #関連情報取得
    elif button_id == 'relate-graph-button':
        # ロードメッセージを設定
        if not selection:
            raise dash.exceptions.PreventUpdate
        loading_message = 'グラフ生成中...'
        print(len(selection))
        selected_nodes = [node for node in net_data['nodes'] if node['id'] in selection['nodes']]
        selected_edges = [edge for edge in net_data['edges'] if edge['from'] in selection['nodes'] or edge['to'] in selection['nodes']]
        target_embeddings=[node['label'] for node in selected_nodes]
        print(target_embeddings)
        # relate_map関数でノードとエッジを関連付け

        new_map = generate_maps_from_reflections(selected_nodes,'情報', 'UECsubject_maps3/', 'output_maps/', user_year, 10, 3)
        # ノードの更新
        new_nodes = {node['id']: node for node in new_map['nodes']}
        for i, node in enumerate(net_data['nodes']):
            print(node)
            if node['id'] in new_nodes:
                net_data['nodes'][i] = new_nodes.pop(node['id'])
        net_data['nodes'].extend(new_nodes.values())
        
        # エッジの更新
        new_edges = {(edge['from'], edge['to']): edge for edge in new_map['edges']}
        for i, edge in enumerate(net_data['edges']):
            edge_tuple = (edge['from'], edge['to'])
            if edge_tuple in new_edges:
                net_data['edges'][i] = new_edges.pop(edge_tuple)
        net_data['edges'].extend(new_edges.values())

        loading_message = ''
        
        return net_data,  loading_message
    
    elif button_id == 'get-info-button':

        if not selection:
            raise dash.exceptions.PreventUpdate
        
        # ロードメッセージを設定
        loading_message = 'グラフ生成中...'
        print(selection)

        # 選択されたノードとエッジを取得し、新しいグラフを生成
        selected_nodes = [node for node in net_data['nodes'] if node['id'] in selection['nodes']]
        selected_edges = [edge for edge in net_data['edges'] if edge['from'] in selection['nodes'] or edge['to'] in selection['nodes']]
        
        # relate_map関数でノードとエッジを関連付け
        new_map = relate_map(selected_nodes, selected_edges)
        
        # ノードの更新
        new_nodes = {node['id']: node for node in new_map['nodes']}
        for i, node in enumerate(net_data['nodes']):
            if node['id'] in new_nodes:
                net_data['nodes'][i] = new_nodes.pop(node['id'])
        net_data['nodes'].extend(new_nodes.values())
        
        # エッジの更新
        new_edges = {(edge['from'], edge['to']): edge for edge in new_map['edges']}
        for i, edge in enumerate(net_data['edges']):
            edge_tuple = (edge['from'], edge['to'])
            if edge_tuple in new_edges:
                net_data['edges'][i] = new_edges.pop(edge_tuple)
        net_data['edges'].extend(new_edges.values())

        loading_message = ''
        
        return net_data,  loading_message
    
    save_user_data(name_input,input_text,net_data)
    # 最終的に更新されたネットワークデータを返却
    return net_data, loading_message

# ノード選択、ポップアップ表示、周辺情報取得、保存機能のコールバック
@app.callback(
    Output('node-popup', 'style'),
    Output('node-info', 'children'),
    Input('net', 'selection'),
    Input('close-button', 'n_clicks'),
    Input('get-info-button', 'n_clicks'),
    Input('reflect-button', 'n_clicks'),
    Input('save-button', 'n_clicks'),
    State('net', 'data'),
    State('name-input', 'value'),
    State('reflection-text', 'value'),
    prevent_initial_call=True
)
def toggle_node_popup(selection, close_clicks, get_info_clicks, reflect_clicks, save_clicks, data, name_input, reflection_text):
    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]

    # ノードが選択されたとき、ポップアップを表示
    if triggered_id == 'net' and selection['nodes']:
        node_id = selection['nodes'][0]
        node = next((node for node in data['nodes'] if node['id'] == node_id), None)
        if node:
            node_info = html.Div([
                html.P(f"選択されたノード: {node['label']}", style={'fontFamily': 'Garamond', 'fontSize': '18px', 'color': '#7D7461'}),
                html.P(f"説明: {node.get('sentence', '説明はありません')}", 
                       style={'fontFamily': 'Garamond', 'fontSize': '16px', 'color': '#7D7461'})
            ])
            return {
                'width': '20%', 'height': 'auto', 'position': 'absolute', 'top': '10%', 
                'right': '0', 'border': '2px dashed #7D7461', 'padding': '10px', 
                'backgroundColor': '#F4EBD9', 'transition': 'right 0.5s cubic-bezier(0.25, 1, 0.5, 1)', 'display': 'block'
            }, node_info
# 保存ボタンが押されたとき
    elif triggered_id == 'save-button' and name_input:
        save_user_data(name_input,reflection_text,data)
        
        return dash.no_update  # ポップアップの状態を変更せず、保存処理のみ実行
    
    # それ以外のボタンが押されたら、ポップアップを非表示
    return {'display': 'none'}, None

if __name__ == '__main__':
    app.run_server(debug=True)
