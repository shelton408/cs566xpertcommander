
class TheGameCard(object):

    rank =[str(i) for i in range(2, 49)]

    def __init__(self, rank):
        ''' Initialize the class of TheGameCard

        Args:
            card_type (str): The type of card
        '''
        self.rank = str(rank)

    def __gt__(self, other):
        return int(self.rank) > int(other.rank)

    def __lt__(self, other):
        return int(self.rank) < int(other.rank)