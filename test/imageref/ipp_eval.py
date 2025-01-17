import re
import argparse
import pandas as pd
import jsonlines
from ast import literal_eval
import numpy as np
import json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_file", default="", type=str)
    parser.add_argument("--label_file", default="", type=str)
    parser.add_argument("--event_file", default="", type=str)
    parser.add_argument("--annotation_file", default="", type=str)
    parser.add_argument("--model_type", default="guided", type=str)
    args = parser.parse_args()
    return args


def bpm(u, matchR, seen, bpGraph):
    """ A DFS based recursive function that returns true if a matching for vertex u is possible """
    for v in range(len(bpGraph[0])):
        if not seen[v] and bpGraph[u][v]:
            seen[v] = True
            if matchR[v] == -1 or bpm(matchR[v], matchR, seen, bpGraph):
                matchR[v] = u
                return True
    return False

def maxBPM(bpGraph):
    """ Returns maximum number of matching from M to N """
    matchR = [-1] * len(bpGraph[0])
    result = 0
    for i in range(len(bpGraph)):
        seen = [False] * len(bpGraph[0])
        if bpm(i, matchR, seen, bpGraph):
            result += 1
    return result


def max_items(N, sets):
    if not sets:
        return 0
    
    # Create a mapping from items (strings) to unique integers
    item_to_index = {}
    index = 0
    
    for s in sets:
        if len(s) == 0:
            continue
        for item in s:
            if item not in item_to_index:
                item_to_index[item] = index
                index += 1

    max_item = len(item_to_index)

    bpGraph = [[0] * max_item for _ in range(N)]
    for i in range(N):
        for item in sets[i]:
            bpGraph[i][item_to_index[item]] = 1
            
    return maxBPM(bpGraph)


def recall_3(labels, results):
    all = max_items(len(labels), [labels[p]['3'] for p in labels])
    
    match = 0
    for p in labels:
        a = labels[p]['3']
        if p in results and results[p] in a:
            match += 1

    if all == 0:
        return 1
    
    return match / all

def recall_1(labels, results):
    all = max_items(len(labels), [list(set(labels[p]['3'] + labels[p]['2'] + labels[p]['1'])) for p in labels])
    match = 0
    for p in labels:
        a = labels[p]['3'] + labels[p]['2'] + labels[p]['1']
        if p in results and results[p] in a:
            match += 1
    if all == 0:
        return 1
    return match / all


def precion(labels, results):
    all, match = 0, 0
    for p in results:
        if results[p] and p in labels:
            all += 1
            if results[p] in labels[p]['3'] + labels[p]['2'] + labels[p]['1']:
                match += 1

    if all == 0:
        return 1
    return match / all


def bpm_score(u, matchR, seen, bpGraph, score):
    """ A DFS based recursive function that returns true if a matching for vertex u is possible. 
        Also tracks the maximum score we can achieve by finding the best matches. """
    for v in range(len(bpGraph[0])):
        if not seen[v] and bpGraph[u][v]:
            seen[v] = True
            # Try to match either v with u, or if v is already matched, try to rematch its current match
            if matchR[v] == -1 or bpm_score(matchR[v], matchR, seen, bpGraph, score):
                matchR[v] = u
                score[u] = max(score[u], bpGraph[u][v])  # Update the score for vertex u
                return True
    return False

def maxBPM_score(bpGraph):
    """ Returns maximum matching score from M to N by finding the optimal matchings. """
    matchR = [-1] * len(bpGraph[0])  # Match result for right side
    score = [0] * len(bpGraph)  # Store the score for each left vertex (insertion positions)
    
    # Try to find a match for every vertex on the left side (insertion positions)
    for i in range(len(bpGraph)):
        seen = [False] * len(bpGraph[0])
        if bpm_score(i, matchR, seen, bpGraph, score):
            pass  # If a match is found, the score is updated in the bpm function
    
    # The result is the sum of the scores of matched positions
    return sum(score)

def max_items_score(N, sets, item_scores):
    if not sets:
        return 0
    
    # Create a mapping from items (strings) to unique integers
    item_to_index = {}
    index = 0
    
    for s in sets:
        for item in s:
            if len(s) == 0:
                continue
            if item not in item_to_index:
                item_to_index[item] = index
                index += 1

    max_item = len(item_to_index)

    # Create the bipartite graph where bpGraph[i][j] represents the score for placing image j at position i
    bpGraph = [[0] * max_item for _ in range(N)]
    
    # For each position, insert the images with their corresponding scores into the bipartite graph
    for position, score_dict in item_scores.items():
        position_index = position  # Assuming position is directly used as index (0-indexed)
        
        # For each score group (1, 2, 3, etc.)
        for score_value, images in score_dict.items():
            for image in images:
                if image in item_to_index:
                    image_index = item_to_index[image]
                    
                    match = re.search(r"<Position (\d+)>", position_index)                
                    position_number = int(match.group(1)) - 1
                    bpGraph[position_number][image_index] = int(score_value)  # Assign the score to the bipartite graph
    # Call the maxBPM function to compute the maximum score achievable by matching
    return maxBPM_score(bpGraph)

def score(labels, results):
    max_score = max_items_score(len(labels), [list(set(labels[p]['3'] + labels[p]['2'] + labels[p]['1'])) for p in labels], labels)
    
    score = 0
    for p in results:
        if results[p] and p in labels:
            if results[p] in labels[p]['3']:
                score += 3
            elif results[p] in labels[p]['2']:
                score += 2
            elif results[p] in labels[p]['1']:
                score += 1
            else:
                score -= 1
    

    if max_score == 0:
        return 1
    
    return score / max_score


def parse_position_map(position_map, image_map):
    for p in position_map:
        for s in position_map[p]:
            position_map[p][s] = [image_map[i] for i in position_map[p][s]]
    return position_map


def load_label(df):
    labels = {}
    for index in df.index:
        eid = df.loc[index, "paasEventID"]
        position_map = df.loc[index, "result"]
        image_map = df.loc[index, "image_map"]
        position_map = parse_position_map(position_map, image_map)
        labels[eid] = position_map, image_map
    return labels

def extract_number(image_str):
    match = re.search(r"IMAGE_(\d+)", image_str)
    if match:
        return int(match.group(1))
    return None

def load_result(result, image_paths):
    extract_result = {}
    predictions = result["predictions"]
    
    already_inserted_img = set()
    for prediction in predictions:
        image = prediction["image"]
        if image != "Not Insert":
            if image in already_inserted_img:
                continue
            already_inserted_img.add(image)
            position = prediction["position"]
            image_number = extract_number(image)
            extract_result[position] = image_paths[image_number-1]
    
    return extract_result
    

def eval_three_stage(df, output_file):
    labels = load_label(df)
    
    results = {}
    with jsonlines.open(output_file, 'r') as reader:
        for obj in reader:
            eid = obj["event_id"]
            try:
                results[eid] = load_result(json.loads(obj["result"]), obj["image_paths"])
            except:
                print("json load result failed. Skip.")
         
    r1, r3, prec, scores = [], [], [], []
    for eid in labels:
        if eid in labels and str(eid) in results:
            label_positions, candidate_images = labels[eid]
            result = results[str(eid)]
            recall_score = recall_1(label_positions, result)
            recall3_score = recall_3(label_positions, result)
            prec_score = precion(label_positions, result)
            s = score(label_positions, result)

            r1.append(recall_score)
            r3.append(recall3_score)
            prec.append(prec_score)
            scores.append(s)
    
    num = len(r1)
    r1 = np.mean(r1)
    r3 = np.mean(r3)
    prec = np.mean(prec)
    scores = np.mean(scores)
    f1 = 2 * prec * r3 / (prec + r3)

    print(f"{num}\{r1:.4f}\{r3:.4f}\{prec:.4f}\{f1:.4f}\{scores:.4f}")    


def extract_filled_positions(text1, text2, event_imgs):
    pattern = r"<Position (\d+)>"
    matches = list(re.finditer(pattern, text1))
    
    filled_positions = {}
    last_end_idx = 0
    img_map = {f"IMAGE_{i+1}": event_imgs[i] for i in range(len(event_imgs))}

    for i, match in enumerate(matches):
        position_id = match.group(0)
        start_idx = match.start()
        end_idx = match.end()

        before_text = text1[last_end_idx:start_idx]
        
        before_in_text2 = text2.find(before_text)
        if before_in_text2 != -1:
            filled_start = before_in_text2 + len(before_text)
            
            if i + 1 < len(matches):
                next_position_start = matches[i + 1].start()
                after_text = text1[end_idx:next_position_start]
            else:
                after_text = ""
            
            after_in_text2 = text2.find(after_text) if after_text else len(text2)
            
            filled_content = text2[filled_start:after_in_text2].strip()
            try:
                filled_positions[position_id] = re.search(r'IMAGE_\d+', filled_content).group() if filled_content != "" else ""
            except:
                continue
                
            last_end_idx = end_idx
    
    already_inserted_img = set()
    
    for p, v in filled_positions.items():
        if v in already_inserted_img:
            filled_positions[p] = None
            continue
        else:
            already_inserted_img.add(v)
            
        if v in img_map:
            filled_positions[p] = img_map[v]
        elif v == "":
            filled_positions[p] = None
        else:
            print(f"{v} not in img_map")
            # import pdb; pdb.set_trace()    
    return filled_positions

def eval_guided(df, output_file, event_file, annotation_file):
    labels = load_label(df)
    
    event_texts = {}
    with jsonlines.open(event_file, 'r') as reader:
        for obj in reader:
            eid = int(obj["paasEventID"])
            if eid in test_eids:
                event_text = preprocess_text(obj["title"], obj["summary"])[0]
                event_texts[eid] = event_text
    
    event_imgs = {}
    with jsonlines.open(annotation_file, 'r') as reader:
        for obj in reader:
            eid = obj["id"]
            event_imgs[eid] = ["/".join(i.split("/")[-2:]) for i in obj["image"]]
    
    r1, r3, prec, scores = [], [], [], []
    with jsonlines.open(output_file, 'r') as reader:
        for obj in reader:
            eid = obj["id"]
            filled_positions = extract_filled_positions(event_texts[eid], obj["result"], event_imgs[eid])
            label_positions, candidate_images = labels[eid]
            
            recall_score = recall_1(label_positions, filled_positions)
            recall3_score = recall_3(label_positions, filled_positions)
            prec_score = precion(label_positions, filled_positions)
            s = score(label_positions, filled_positions)
            # import pdb; pdb.set_trace()
            
            r1.append(recall_score)
            r3.append(recall3_score)
            prec.append(prec_score)
            scores.append(s)
    
    num = len(r1)
    r1 = np.mean(r1)
    r3 = np.mean(r3)
    prec = np.mean(prec)
    scores = np.mean(scores)
    f1 = 2 * prec * r3 / (prec + r3)

    print(f"{num}\{r1:.4f}\{r3:.4f}\{prec:.4f}\{f1:.4f}\{scores:.4f}")               
            

def preprocess_text(title: str, text: str):
    lines = [f"# {title}"] + [
        i.strip() for i in text.strip().split("\n") if i.strip() != ""
    ]
    output_lines = []
    num_positions = 0

    for line in lines:
        output_lines.append(line)
        if not line.startswith("#"):
            output_lines.append(f"<Position {num_positions + 1}>")
            num_positions += 1

    processed_text = "\n\n".join(output_lines)
    return processed_text, num_positions

if __name__ == "__main__":
    args = parse_args()
    df = pd.read_csv(
        args.label_file,
        converters={'image_map': literal_eval, 'result': literal_eval}
    )
    
    test_eids = set(df["paasEventID"].values.tolist())
        
    if args.model_type == "three_stage":
        eval_three_stage(df, args.output_file)
    elif args.model_type == "guided":
        eval_guided(df, args.output_file, args.event_file, args.annotation_file)
