import argparse


class ArgParsing:
    gamesNames = ['Breakout', 'Pong', 'Seaquest', 'MsPacman']

    def __int__(self):

        self.parser = argparse.ArgumentParser(description="Options for training")

        self.parser.add_argument('-g',type=str)
        self.parser.add_argument('-pxl')
        self.parser.add_argument('-e', type=int, default=10000)
        self.parser.add_argument('-r',
                                 help='Train while rendering (not recommended if you are actually training a model)')
        self.parser.add_argument('-s', help='Save models every e episodes')

        self.args = self.parser.parse_args()

        validGame = False

        for i in range(len(self.gamesNames)):
            if self.args.g == self.gamesNames[i]:
                validGame = True

        if not validGame: print("Please enter a valid game")

        print(validGame)

    def get_args(self):
        return self.args
