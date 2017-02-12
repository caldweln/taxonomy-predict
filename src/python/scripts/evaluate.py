
#
# return Precision, Recall & F-score foreach pair (prediction,target)
#
def score_predictions(preds,targets):
    if len(preds) != len(targets):
        raise ValueError("predictions and targets have unequal lengths")
    return list(map(score_class_pred_pair, list(zip(preds,targets))))

def score_class_pred_pair(pred_targ):
    return score_class_pred(*pred_targ)
    
#
# calculate Precision, Recall & F-score
#
# using LCA augmentation for hierarhcical classes (https://arxiv.org/abs/1306.6802)
#
def score_class_pred(pred,targ):
    pred_aug,targ_aug = get_augmented_classes(pred,targ)
    intersection = set(pred_aug).intersection(set(targ_aug))

    p_score = score_class(intersection) / score_class(pred_aug)
    r_score = score_class(intersection) / score_class(targ_aug)

    f_score = 0
    if p_score != 0 and r_score != 0:
        f_score  = (2*p_score*r_score) / (p_score+r_score)

    return [p_score, r_score, f_score]

#
# left trim classes up to lowest common ancestor (LCA)
#
def get_augmented_classes(pred,targ):
    for lca in range(0,(len(pred))):
        if lca >= len(targ):
            break
        if pred[lca] != targ[lca]:
            break
    lca = lca-1
    pred_aug = pred[lca:]
    targ_aug = targ[lca:]
    return (pred_aug,targ_aug)

#
# simple node count
#
def score_class(c):
    return len(c)
