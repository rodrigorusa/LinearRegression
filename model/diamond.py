class Diamond:
    id = 0
    carat = 0
    cut = 0
    color = 0
    clarity = 0
    depth = 0
    table = 0
    x = 0
    y = 0
    z = 0
    price = 0


    def defineCutValue(self, cut):
        switchCut = {
            "Fair": 1,
            "Good": 2,
            "Very Good": 3,
            "Premium": 4,
            "Ideal": 5
        }

        return switchCut.get(cut, 0)

    def defineColorValue(self, color):
        switchColor = {
            "J": 1,
            "I": 2,
            "H": 3,
            "G": 4,
            "F": 5,
            "E": 6,
            "D": 7
        }

        return switchColor.get(color, 0)

    def defineClarityValue(self, clarity):
        switchClarity = {
            "I3": 1,
            "I2": 2,
            "I1": 3,
            "SI2": 4,
            "SI1": 5,
            "VS2": 6,
            "VS1": 7,
            "VVS2": 8,
            "VVS1": 9,
            "IF": 10,
            "FL": 11
        }

        return switchClarity.get(clarity, 0)

    def __init__(self, id, carat, cut, color, clarity, depth, table, x, y, z, price):
        self.id = int(id)
        self.carat = float(carat)
        self.cut = self.defineCutValue(cut)
        self.color = self.defineColorValue(color)
        self.clarity = self.defineClarityValue(clarity)
        self.depth = float(depth)
        self.table = float(table)
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.price = float(price)

    def __repr__(self):
        return "carat: {0} cut: {1} color: {2} clarity: {3} depth: {4} table: {5} x: {6} y: {7} z: {8} price: {9}"\
                .format(self.carat, self.cut, self.color, \
                        self.clarity, self.depth, self.table, \
                        self.x, self.y, self.z, self.price)
