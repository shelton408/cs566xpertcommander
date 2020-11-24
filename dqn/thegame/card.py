class TheGameCard(object):

    def __init__(self, rank):
        ''' Initialize the class of TheGameCard

        Args:
            rank (str): The number of the card
        '''
        self.rank = str(rank)

    def __lt__(self, other):
        return int(self) < int(other)

    def __gt__(self, other):
        return int(self) > int(other)

    def __eq__(self, other):
        return int(self) == int(other)

    def __ne__(self, other):
        return not self == other

    def __str__(self):
        return self.rank

    def __int__(self):
        return int(self.rank)
