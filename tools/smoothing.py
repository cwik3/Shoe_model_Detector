
from collections import deque

class StableLabel:
    def __init__(self, maxlen=30, show_thr=0.70, hide_thr=0.55, stick_frames=10):
        self.labels = deque(maxlen=maxlen)  # holds (label, conf)
        self.last_stable = None             # (label, conf) or None
        self.sticky = 0
        self.show_thr = show_thr
        self.hide_thr = hide_thr
        self.stick_frames = stick_frames

    def update(self, label, conf):
        if label is None:
            # no candidate this frame; decay stickiness
            self.sticky = max(0, self.sticky - 1)
            if self.sticky == 0:
                self.last_stable = None
            return self.last_stable

        self.labels.append((label, conf))

        # majority by count, tie-breaker by avg conf
        counts, confs = {}, {}
        for l, c in self.labels:
            counts[l] = counts.get(l, 0) + 1
            confs[l]  = confs.get(l, 0)  + c
        avg_conf = {l: confs[l]/counts[l] for l in counts}
        w_label = max(avg_conf.keys(), key=lambda l: (counts[l], avg_conf[l]))
        w_conf  = avg_conf[w_label]

        if self.last_stable is None:
            if w_conf >= self.show_thr:
                self.last_stable = (w_label, w_conf)
                self.sticky = self.stick_frames
        else:
            if w_label == self.last_stable[0]:
                self.last_stable = (w_label, max(self.last_stable[1], w_conf))
                self.sticky = self.stick_frames
            else:
                if w_conf >= self.show_thr:
                    self.last_stable = (w_label, w_conf)
                    self.sticky = self.stick_frames
                else:
                    self.sticky = max(0, self.sticky - 1)
                    if self.sticky == 0 and w_conf < self.hide_thr:
                        self.last_stable = None

        return self.last_stable
