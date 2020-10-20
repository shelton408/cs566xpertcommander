

class Deck:
    def __init__(self, ascending=True):
        self.top_card = 1 if ascending else 100
        self.isAscending = ascending
        self.isDecending = not ascending
