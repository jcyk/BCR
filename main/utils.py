# for easy computation
class UpdateMatrix:
    def __init__(self, yy=0, yn=0, ny=0, nn=0):
        self.yy = yy
        self.yn = yn
        self.ny = ny
        self.nn = nn
    
    def copy(self):
        return UpdateMatrix(self.yy, self.yn, self.ny, self.nn)

    def update(self, old_yes, new_yes):
        if old_yes:
            if new_yes:
                self.yy += 1
            else:
                self.yn += 1
        else:
            if new_yes:
                self.ny += 1
            else:
                self.nn += 1
        
    def aggregate(self, another_matrix):
        self.yy += another_matrix.yy
        self.yn += another_matrix.yn
        self.ny += another_matrix.ny
        self.nn += another_matrix.nn
    def __repr__(self):
        return f"yy: {self.yy} yn: {self.yn} ny: {self.ny} nn: {self.nn}"
    
    @property
    def tot(self):
        return self.yy + self.yn + self.ny + self.nn
    
    @property
    def nfr(self):
        return self.yn/self.tot
    
    @property
    def pfr(self):
        return self.ny/self.tot
    
    @property
    def old_acc(self):
        return (self.yy + self.yn) /self.tot
    
    @property
    def new_acc(self):
        return (self.yy + self.ny) /self.tot
    
    @property
    def nfi(self):
        if self.new_acc == 1.:
            return 0
        return self.nfr / (1.0 - self.new_acc)
    @property
    def any(self):
        return (self.yy + self.ny + self.yn)
    
    def report(self):
        print (f"NFR: {self.nfr*100:.2f}% NFI: {self.nfi*100:.2f}% new Acc: {self.new_acc*100:.2f}% PFR: {self.pfr*100:.2f}% old Acc: {self.old_acc*100:.2f}% count: {self.tot}")

    def report_table(self, head):
        print (f"{head}#{self.nfr*100:.2f}%#{self.pfr*100:.2f}%#{self.old_acc*100:.2f}%#{self.new_acc*100:.2f}%#{self.nfi*100:.2f}%#{self.tot}")

class Statistics:
    def __init__(self):
        self.data = {}
    
    def update(self, key, old_yes, new_yes):
        if key not in self.data:
            self.data[key] = UpdateMatrix()
        self.data[key].update(old_yes, new_yes)

    def aggregate(self, another_statistics):
        for key in another_statistics.keys:
            if key in self.data:
                self.data[key].aggregate(another_statistics[key])
            else:
                self.data[key] = another_statistics[key].copy()

    def __getitem__(self, key):
        return self.data[key]

    @property
    def keys(self):
        return self.data.keys()
    
    def report(self, keys=[]):
        for x in self.keys:
            show = True if keys is None else False
            for y in keys:
                if x.startswith(y):
                    show = True
            if show:
                print (x)
                self.data[x].report()

def merge_statistics(stats):
    stat = Statistics()
    for stat_i in stats:
        stat.aggregate(stat_i)
    return stat
    