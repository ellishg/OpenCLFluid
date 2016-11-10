#!/usr/bin/env python

import matplotlib.pyplot as plt

class DeviceRunningTime:

    def __init__(self, n):
        self.name = n
        self.sim_size = []
        self.add_event = []
        self.add_source = []
        self.set_bnd = []
        self.diffuse = []
        self.advect = []
        self.project_a = []
        self.project_b = []
        self.project_c = []
        self.runtime = []
        self.total_runtime = []

    def set_runtimes(self, size, ade, ads, sb, d, a, pa, pb, pc, r, tr):
        self.sim_size.append(size)
        self.add_event.append(ade)
        self.add_source.append(ads)
        self.set_bnd.append(sb)
        self.diffuse.append(d)
        self.advect.append(a)
        self.project_a.append(pa)
        self.project_b.append(pb)
        self.project_c.append(pc)
        self.runtime.append(r)
        self.total_runtime.append(tr)

    def plot(self):
        plt.plot(self.sim_size, self.total_runtime)


macbook = DeviceRunningTime("MacBook GPU")
macbook.set_runtimes(64, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15)
macbook.set_runtimes(128, 0, 0, 0, 0, 0, 0, 0, 0, 0, 20)
macbook.set_runtimes(256, 0, 0, 0, 0, 0, 0, 0, 0, 0, 29)
macbook.set_runtimes(512, 0, 0, 0, 0, 0, 0, 0, 0, 0, 68)
macbook.set_runtimes(1024, 0, 0, 0, 0, 0, 0, 0, 0, 0, 258)
macbook.set_runtimes(2048, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3438)

iMac = DeviceRunningTime("iMac GPU")
iMac.set_runtimes(64, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1)
iMac.set_runtimes(128, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2)
iMac.set_runtimes(256, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2)
iMac.set_runtimes(512, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6)
iMac.set_runtimes(1024, 0, 0, 0, 0, 0, 0, 0, 0, 0, 28)
iMac.set_runtimes(2048, 0, 0, 0, 0, 0, 0, 0, 0, 0, 348)

plt.figure(1)

macbook.plot()
iMac.plot()

plt.show()
