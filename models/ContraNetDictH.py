import logging
import torch
from models.ContraNet2dist import ContraNet2dist
from models.ContraNetDict import ContraNetDict


class ContraNetDictH(ContraNetDict):
    def __init__(self, classifier, cls_norm, hard_drop):
        super(ContraNetDictH, self).__init__(classifier, cls_norm)
        self.hard = hard_drop
        logging.info("[detectorDict] hard_drop={}".format(self.hard))

    def get_thresh(self, test_loader, drop_rate=[0.05, 0.05]):
        thresh = []
        pass_prev = None
        hard_idx = 0
        drop_rate_idx = 0
        for d_name in self.detector:
            if isinstance(self.detector[d_name], ContraNet2dist):
                torch.random.manual_seed(1)  # manual seed after D and DM
                out_list = self.detector[d_name].get_judge_results(test_loader)
                if pass_prev is None:
                    logging.warn(
                        "No previous at {}, you may check".format(d_name))
                else:
                    hard = self.hard[hard_idx]
                    out_sorted, inds = self.detector[d_name].sort_results(
                        out_list)  # need: reject ..|... pass
                    pass_prev_sorted = pass_prev[inds]
                    # this is the upper bound of this drop only.
                    prev_drop_rate = (
                        (pass_prev == 0).sum() / len(pass_prev)).item()
                    if prev_drop_rate >= hard:
                        raise RuntimeError("This is no space to drop fot this.")
                    logging.info(
                        "[detectorDict] prev drop rate is {}, continue.".format(
                            prev_drop_rate))
                    stop_num_single = int(
                        len(out_list) * (hard - prev_drop_rate))
                    pass_single = 0
                    for idx, this in enumerate(out_sorted):
                        if pass_single < stop_num_single:
                            if pass_prev_sorted[idx]:
                                # only count on those passed before
                                pass_single += 1
                                pass_prev_sorted[idx] = 0
                        else:
                            thresh.append(this.item())
                            break
                    _, back_inds = inds.sort()
                    pass_prev = pass_prev_sorted[back_inds]
                hard_idx += 1
            else:
                this_thresh, out_list = self.detector[d_name].get_thresh(
                    test_loader, drop_rate=drop_rate[drop_rate_idx],
                    return_list=True)
                if pass_prev is None:
                    pass_prev = self.detector[d_name].judge_mlp_thresh(
                        out_list, this_thresh)
                else:
                    pass_prev = torch.logical_and(
                        pass_prev, self.detector[d_name].judge_mlp_thresh(
                            out_list, this_thresh))
                thresh.append(this_thresh)
                drop_rate_idx += 1
            logging.info("[detectorDict] threshold for {} = {}".format(
                d_name, thresh[-1]))
        return thresh
