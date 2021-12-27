import random
import GameData
from copy import deepcopy


class Card(object):
    def __init__(self) -> None:
        super().__init__()
        self.color = ["red", "yellow", "green", "blue", "white"]
        self.value = [1, 2, 3, 4, 5]

    def assign_color(self, c):
        self.color = [c]

    def remove_color(self, c):
        if len(self.color) > 1:
            if c in self.color:
                self.color.remove(c)

    def assign_value(self, v):
        self.value = [v]

    def remove_value(self, v):
        if len(self.value) > 1:
            if v in self.value:
                self.value.remove(v)

    def is_playable(self, tableCards):
        if len(self.value) > 1:
            return False
        for c in self.color:
            if len(tableCards[c]) != self.value[0] - 1:
                return False
        return True                


class OtherPlayer(object):
    def __init__(self, id, num_players) -> None:
        super().__init__()
        self.id = id
        self.num_players = num_players
        self.cards = []
        if num_players < 4:
            self.n_cards = 5
        else:
            self.n_cards = 4
        for _ in range(self.n_cards):
            self.cards.append(Card())

    def remove_card(self, pos, played_cards):
        del self.cards[pos]
        if played_cards <= 50:
            self.cards.append(Card())
        else:
            self.n_cards -= 1

    def receive_hint(self, t, value, positions):
        if t == "color":
            for p in range(self.n_cards):
                if p in positions:
                    self.cards[p].assign_color(value)
                else:
                    self.cards[p].remove_color(value)

        elif t == "value":
            for p in range(self.n_cards):
                if p in positions:
                    self.cards[p].assign_value(value)
                else:
                    self.cards[p].remove_value(value)

    def score_hint(self, t, value, positions, state):
        score = 0

        cards_copy = deepcopy(self.cards)

        if t == "color":
            for p in range(self.n_cards):
                if p in positions:
                    score += len(cards_copy[p].color) - 1
                    cards_copy[p].assign_color(value)
                else:
                    if value in cards_copy[p].color:
                        score += 1
                        cards_copy[p].remove_color(value)

        elif t == "value":
            for p in range(self.n_cards):
                if p in positions:
                    score += len(cards_copy[p].value) - 1
                    cards_copy[p].assign_value(value)
                else:
                    if value in cards_copy[p].value:
                        score += 1
                        cards_copy[p].remove_value(value)

        if [i for i in range(len(cards_copy)) if cards_copy[i].is_playable(state.tableCards)]:
            return 100
        return score


class Player(object):
    def __init__(self, id, num_players) -> None:
        super().__init__()
        self.id = id
        self.num_players = num_players
        self.cards = []
        if num_players < 4:
            self.n_cards = 5
        else:
            self.n_cards = 4
        for _ in range(self.n_cards):
            self.cards.append(Card())
        
        self.other_players = [OtherPlayer(i, num_players) for i in range(num_players) if i != id]

    def compute_played_cards(self, state):
        played_cards = self.n_cards + len(state.discardPile) 
        for p in state.players:
            played_cards += len(p.hand)
        for pos in state.tableCards:
            played_cards += len(state.tableCards[pos])
        return played_cards

    def play(self, state):
        
        playable_cards = [i for i in range(self.n_cards) if self.cards[i].is_playable(state.tableCards)]

        if len(playable_cards) > 0: # play if possible
            action = "play"
            cardOrder = playable_cards[0]

        elif state.usedNoteTokens == 8: # forced to discard
            action = "discard"

        elif state.usedNoteTokens == 0: # forced to hint
            action = "hint"

        else: # if there are tokens available, hint -> maybe modify with something more sophisticated
            action = "hint"

        played_cards = self.compute_played_cards(state)

        if action == "discard":
            cardOrder = random.randint(0, self.n_cards - 1)
            if [i for i in range(self.n_cards) if 5 not in self.cards[i].value]: # at least one card that is not a 5
                while 5 in self.cards[cardOrder].value:
                    cardOrder = random.randint(0, self.n_cards - 1)
            del self.cards[cardOrder]
            if played_cards < 50:
                self.cards.append(Card())
            else:
                self.n_cards -= 1
            action = GameData.ClientPlayerDiscardCardRequest(str(self.id), cardOrder)

        elif action == "play":
            del self.cards[cardOrder]
            if played_cards < 50:
                self.cards.append(Card())
            else:
                self.n_cards -= 1
            action = GameData.ClientPlayerPlayCardRequest(str(self.id), cardOrder)

        elif action == "hint":

            best_score = 0
            best_t = ""
            best_val = None
            best_dest = 0
            for p in state.players:
                # find player in local list
                for loc_p in self.other_players:
                    if str(loc_p.id) == p.name:
                        break
                # try color hint
                available_colors = []
                for c in p.hand:
                    if c.color not in available_colors:
                        available_colors.append(c.color)
                for c in available_colors:
                    score = loc_p.score_hint("color", c, [i for i in range(len(p.hand)) if p.hand[i].color == c], state)
                    if score > best_score:
                        best_score = score
                        best_t = "color"
                        best_val = c
                        best_dest = p.name
                # try value hint
                available_values = []
                for c in p.hand:
                    if c.value not in available_values:
                        available_values.append(c.value)
                for v in available_values:
                    score = loc_p.score_hint("value", v, [i for i in range(len(p.hand)) if p.hand[i].value == v], state)
                    if score > best_score:
                        best_score = score
                        best_t = "value"
                        best_val = v
                        best_dest = p.name

            for loc_p in self.other_players:
                if str(loc_p.id) == best_dest: # WRONG!!!
                    loc_p.receive_hint(best_t, best_val, [i for i in range(len(p.hand)) if p.hand[i].value == best_val])
                    break
            action = GameData.ClientHintData(str(self.id), best_dest, best_t, best_val)

        return action

    def receive_hint(self, t, value, positions):
        if t == "color":
            for p in range(self.n_cards):
                if p in positions:
                    self.cards[p].assign_color(value)
                else:
                    self.cards[p].remove_color(value)

        elif t == "value":
            for p in range(self.n_cards):
                if p in positions:
                    self.cards[p].assign_value(value)
                else:
                    self.cards[p].remove_value(value)

    def update_other_players(self, data, state):
        if type(data) == GameData.ClientPlayerDiscardCardRequest:
            for loc_p in self.other_players:
                if str(loc_p.id) == data.sender:
                    break
            loc_p.remove_card(data.handCardOrdered, self.compute_played_cards(state))
        
        elif type(data) == GameData.ClientPlayerPlayCardRequest:
            for loc_p in self.other_players:
                if str(loc_p.id) == data.sender:
                    break
            loc_p.remove_card(data.handCardOrdered, self.compute_played_cards(state))
        
        elif type(data) == GameData.ServerHintData:
            if int(data.destination) == self.id:
                self.receive_hint(data.type, data.value, data.positions)
            else:
                for loc_p in self.other_players:
                    if str(loc_p.id) == data.sender:
                        break
                loc_p.receive_hint(data.type, data.value, data.positions)


