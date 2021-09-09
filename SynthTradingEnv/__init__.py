from SynthTradingEnv.SynthCommons import norm_features, load_json, load_npz
import numpy as np
from scipy import stats
import pickle
import math
import matplotlib.pyplot as plt
from random import randint, choice
import gym
from gym import spaces
from gym.utils import seeding
from deprecated import deprecated
import curses
from curseXcel import Table
import torch
import logging
from datetime import datetime

# FOR USE WITH CUPY
# mempool = np.get_default_memory_pool()
# with np.cuda.Device(0):
#     mempool.set_limit(size=10*2*1024**3)  # 20 GiB


# POSITION TYPES
POSTYPE_LONG = 1
POSTYPE_SHORT = 2
POSTYPE_FLAT = 3

# ORDER ACTIONS
# EnterLong() generates OrderAction.Buy
# ExitLong() generates OrderAction.Sell
# EnterShort() generates OrderAction.SellShort
# ExitShort() generates OrderAction.BuyToCover
ORDER_ACTION_BUY = 1
ORDER_ACTION_SELL = 2
ORDER_ACTION_SELL_SHORT = 3
ORDER_ACTION_BUY_TO_COVER = 4

# ORDER TYPES
# OrderType.Limit
# OrderType.Market
# OrderType.MIT
# OrderType.StopMarket
# OrderType.StopLimit
ORDER_TYPE_LIMIT = 1
ORDER_TYPE_MARKET = 2
ORDER_TYPE_STOP_MARKET = 3
ORDER_TYPE_STOP_LIMIT = 4


class TradingEnv(gym.Env):
    """Custom Trading Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 featureList=None,
                 min_margin=None,
                 position_max_qty=10,
                 time_steps=1,
                 start_time_step=0,
                 end_time_step=None,
                 tick_size=0.005,
                 tick_value=25,
                 commission_per_order=2.31,
                 take_profit=1e10,
                 take_loss=1e10,
                 exit_net_drawdown=None,
                 max_time_steps=0,
                 shuffle_per_tran=False,
                 shuffle_per_tran_tt_done=None,
                 normalize_return=True,
                 train_side=None,
                 skip_steps=1,
                 skip_incentive=0,
                 same_entry_penalty=-1000,
                 reward_for_each_step=False,
                 mask_min_steps=False,
                 mask_pos_risk_reward=False,
                 mask_pos_risk_reward_risk=1500,
                 mask_pos_risk_reward_reward=250,
                 mask_flip_flop=False,
                 mask_flip_on_previous_loss=False,
                 mask_wall_value=1000,
                 num_feature_headers=6,
                 data_file_path='C:/Users/jortu/Documents/1Syntheticks-AI/',
                 data_file_name='SI-09-20-FEATURELIST.json',
                 output_path='C:/Users/jortu/Documents/1Syntheticks-AI/',
                 load_historic_rewards=False):
        super(TradingEnv, self).__init__()

        # Environment Name
        self.env = "Sytheticks Trading Env 1.0"

        self.epsilon: float = 1e-6

        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )
        # print(self.device)

        # this will return the difference gained between step i-1 and step i. That way
        # true returns are fed back to Agent on each step.
        self.reward_for_each_step = reward_for_each_step
        # This is the penalty received for entering same side as previous while previous being
        # a negative return.
        self.same_entry_penalty = same_entry_penalty
        # make returns between 0 - 1
        # 0 is short, 1 is long
        self.train_side = train_side
        # used when skipping while training a single side
        self.skip_steps = skip_steps
        # if skipping what incentive to use
        self.skip_incentive = skip_incentive

        # Has the file path and file name of the data for the trading environment.
        self.data_file_name = data_file_name
        self.data_file_path = data_file_path
        self.data_file_path_name = self.data_file_path + self.data_file_name
        self.output_path = output_path

        # Shuffle the starting point
        self.shuffle_per_tran = shuffle_per_tran
        self.shuffle_per_tran_trigger = False
        self.shuffle_per_tran_tt_done = shuffle_per_tran_tt_done
        self.shuffle_per_tran_tt_counter = 0
        self.shuffle_per_tran_step_list = []

        # Used for getting valid order_action masks
        self.mask_min_steps = mask_min_steps
        self.mask_pos_risk_reward = mask_pos_risk_reward
        self.mask_pos_risk_reward_risk = mask_pos_risk_reward_risk
        self.mask_pos_risk_reward_reward = mask_pos_risk_reward_reward
        self.mask_pos_risk_reward_target_hit = False
        self.mask_flip_flop = mask_flip_flop
        self.mask_flip_on_previous_loss = mask_flip_on_previous_loss
        self.mask_wall_value = mask_wall_value
        # This is the initial money balance with we start
        self.balance = 0
        # This is used in normalization of featureList. It will create a "Window" of n time_steps for each
        # row. This is inherited from LSTM, but not really used here.
        self.time_steps = time_steps
        # Starting point of the series
        self.start_time_step = start_time_step
        self.end_time_step = end_time_step
        # Keeps track of the current Time Step we are in at any moment
        self.current_time_step = start_time_step
        # The min amount that a futures instruments moves up and down. Example 0.005 in SI (silver) futures
        self.tick_size = tick_size
        # What 1 tick is valued at in money. Example $25.00 for 1 tick in SI (silver) futures
        self.tick_value = tick_value
        self.point_value = tick_value / self.tick_size
        # Value of the commission charged per order per side (enter or exit) by the futures broker
        self.commission_per_order = commission_per_order
        # Scaler to be stored for normalizing live market data feed.
        self.scaler = None
        # stores how many current time steps of inactivity
        self.inactive_time_step = 0
        # stores how many times steps are currently held in a position
        self.position_time_step = 0
        # stores + or - incentives
        self.incentives = 0
        # This is the min $ margin that is needed to be maintained in an account for a certain instrument
        # SI for example.
        self.min_margin = min_margin
        # This hold previous state. Not currently being used though.
        self.previous_state = []
        # Transactions is where we store all of the transactions (enter/exit pairs)
        self.transaction_dim = 13
        self.transaction_added = False
        # TRANSACTIONS:
        # 0 - pos_type
        # 1 - in_time_step
        # 2 - last_price - --> NOT USED
        # 3 - in_price
        # 4 - out_time_step
        # 5 - last_price - --> NOT USED
        # 6 - out_price
        # 7 - num_time_steps
        # 8 - pnl - --> NOT USED
        # 9 - net_pnl
        # 10 - ticks
        # 11 - commissions
        # 12 - drawdown
        self.transactions = np.empty(shape=(0, self.transaction_dim))

        # Executions: Note that each execution updates the position at the very end of the array.
        # - Time Step
        # - Action (Buy / Sell)
        # - Quantity
        # - Fill Price
        # - Position Type (Long/Short)
        # - Position Qty
        # - Position Avg Price
        self.executions = np.empty(shape=(0, 7))
        self.execution_order_reward = 0
        self.position_max_qty = position_max_qty

        self.reward_total = 0
        self.reward_steps = np.array([[0, 0]])

        # ORDER QUEUE
        # - Order Action
        # - Order Type
        # - Order Price
        # - Order Qty
        # - In Time Step
        # - Is Live Until Canceled (1 / 0). If 0 then order_queue_ts_alive
        self.order_queue = np.empty(shape=(0, 6))
        self.order_queue_is_market = 1
        self.order_queue_ts_alive = 100

        # This contains all of the feature data. First columns include price (last, bid, ask) and
        # how many seconds left to terminate trading session. This data will be extracted form featureList.
        # The clean normalized features are included in feature_info
        self.featureList = featureList
        # This is the drawdown accumulated within a transaction by the difference of max exit_unrlz and current_price
        self.exit_net_drawdown = exit_net_drawdown
        self.max_exit_unrlz = 0
        # This holds the maximum drawdown (loss or potential loss) throughout the episode
        self.max_drawdown = 0
        self.max_accum_reward = 0
        self.max_accum_reward_low_point = 0
        self.max_accum_reward_start = 0
        self.max_accum_reward_low_point_end = 0
        self.max_accum_losses = np.array([[0, 0, 0, 0]])
        # This indicates if environment should take profit at a certain target
        self.take_profit = take_profit
        # This indicates if environment should take loss at a certain target
        self.take_loss = take_loss
        # This holds the max number of time steps. Possibly useful to give sense of termination
        # to be fed back to nn.
        self.max_time_steps = max_time_steps
        # Used for printing transactions if we are on the last part of the session
        self.last_session_printed = False
        # Keeps session #.
        self.session_count = 0
        # Keeps the info regarding whether the state is done or not.
        self.done = False

        # If we want to normalize returns sent back to (-1,+1), then we need to divide by a # that is the MAX expected
        # return of a single transactions. Don't expect clipping to -1, 1 would produce the same effect.
        self.max_return_norm = 20000

        self.episode_rewards = []

        # Number of times transactions have been printed out.
        self.num_print_trans = 0

        if load_historic_rewards:
            self.load_episode_rewards()

        # FROM HERE BELOW BEING ADAPTED FOR GYM
        self.np_random = None

        # Load Feature List
        if self.featureList is None:
            self.load_feature_list(file_path=self.data_file_path_name)

        if self.end_time_step is not None and self.end_time_step > self.start_time_step:
            self.featureList = self.featureList[0:self.end_time_step]

        self.num_featureList_features = self.featureList.shape[1]
        # Features included in the front of each row.
        # Last Price, Bid, Ask, Seconds left in session and Date Time.
        self.num_feature_headers = num_feature_headers
        # Features added by get_state()
        self.num_extra_features = 4
        self.num_features = self.num_featureList_features + self.num_extra_features - self.num_feature_headers

        # 0 -> short
        # 1 -> long
        # 2 -> close position
        # 3 -> do nothing
        self.num_actions = 4

        # ****** Gym specific ******
        self.name = "Syntheticks Trading Environment"
        self.action_space = spaces.Discrete(self.num_actions)
        max_val = np.finfo(np.float32).max
        high = np.full((self.num_features,), max_val)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.seed()
        self.state = None
        self.unwrapped.reward_range = (-float('inf'), float('inf'))
        self.unwrapped.reward_threshold = float('inf')
        self.unwrapped.trials = 100
        self._max_episode_steps = float('inf')

        # FROM HERE UP BEING ADAPTED FOR GYM

        if self.featureList is not None and self.featureList.shape[1]:
            self.feature_info, self.price_info, self.scaler = self.set_and_save_scaler(self.featureList,
                                                                                       'scaler.pickle')
            self.max_time_steps = self.get_max_timesteps()
            assert self.feature_info.shape[1] + self.num_extra_features == self.num_features

        # SETUP CURSES FOR OUTPUT DISPLAY

        # self.stdscr = curses.initscr
        # curses.noecho()
        # curses.cbreak()
        # self.stdscr.keypad(False)
        # curses.wrapper(self.main_curses)
        # curses.nocbreak()
        # stdscr.keypad(False)
        # curses.echo()
        # curses.endwin()

        # rows = 10
        # cols = 20
        # cell = 5
        # width = 20
        # height = 30
        # col_names = True
        # spacing = 1
        # self.table = Table(self.stdscr, rows, cols, cell, width, height, col_names=col_names, spacing=spacing)  # the last two are optional
        # self.set_table_col_names()

    # def main_curses(stdscr):
    #     x = 0
    #     while (x != 'q'):
    #         stdscr.refresh()
    #         x = stdscr.getkey()

    def get_step_for_date(self, step_date):
        for i in range(self.featureList.shape[0]):
            if str(int(self.featureList[i, 4])) == step_date and self.featureList[i, 5] == 0:
                return i

    def list_dates(self):
        for i in range(self.featureList.shape[0]):
            if self.featureList[i, 5] == 0:
                print("Session Start -> " + str(int(self.featureList[i, 4]))[:8] + " - " + str(i))
                logging.debug("Session Start -> " + str(int(self.featureList[i, 4]))[:8] + " - " + str(i))
            elif self.featureList[i, 5] == 1:
                print("Session End -> " + str(int(self.featureList[i, 4]))[:8] + " - " + str(i))
                logging.debug("Session End -> " + str(int(self.featureList[i, 4]))[:8] + " - " + str(i))

    def validate_price_info(self):
        bid_count = 0
        ask_count = 0
        equal_count = 0
        sum_bids = 0
        sum_asks = 0
        adj = 2
        print("Validating BID/ASK tick differences...")
        logging.debug("Validating BID/ASK tick differences...")
        for i in range(self.price_info.shape[0]):
            if self.price_info[i, 0] == self.price_info[i, 1] == self.price_info[i, 2]:
                if i >= self.price_info.shape[0] - 1:
                    self.price_info[i, 1] = self.price_info[i, 0] - adj * self.tick_size
                    self.price_info[i, 2] = self.price_info[i, 0] + adj * self.tick_size
                else:
                    posible_adjs = [0, 1, 2]
                    bid_probs = [0, .4, .6]
                    ask_probs = [0, .4, .6]
                    if self.price_info[i+1, 0] > self.price_info[i, 0]:
                        bid_probs = [.1, .5, .4]
                    elif self.price_info[i+1, 0] < self.price_info[i, 0]:
                        ask_probs = [.1, .5, .4]
                    bid_adj = np.random.choice(posible_adjs, p=bid_probs)
                    ask_adj = np.random.choice(posible_adjs, p=ask_probs)
                    self.price_info[i, 1] = self.price_info[i, 0] - bid_adj * self.tick_size
                    self.price_info[i, 2] = self.price_info[i, 0] + ask_adj * self.tick_size

                equal_count += 1

            else:
                bid_ticks = abs(self.price_info[i, 0] - self.price_info[i, 1]) / self.tick_size
                ask_ticks = abs(self.price_info[i, 0] - self.price_info[i, 2]) / self.tick_size
                if bid_ticks > adj:
                    self.price_info[i, 1] = self.price_info[i, 0] - adj * self.tick_size
                    sum_bids += bid_ticks
                    bid_count += 1
                    # print(abs(self.price_info[i, 0] - self.price_info[i, 1]) / self.tick_size)
                if ask_ticks > adj:
                    self.price_info[i, 2] = self.price_info[i, 0] + adj * self.tick_size
                    sum_asks += ask_ticks
                    ask_count += 1
                    # print(abs(self.price_info[i, 0] - self.price_info[i, 2]) / self.tick_size)
        print("equal_count {}".format(equal_count))
        logging.debug("equal_count {}".format(equal_count))
        print("ask_count {}".format(ask_count))
        logging.debug("ask_count {}".format(ask_count))
        print("ask_count {}".format(ask_count))
        logging.debug("ask_count {}".format(ask_count))
        if bid_count > 0:
            print("bid avg {}".format(sum_bids / bid_count))
            logging.debug("bid avg {}".format(sum_bids / bid_count))
        if ask_count > 0:
            print("ask avg {}".format(sum_asks / ask_count))
            logging.debug("ask avg {}".format(sum_asks / ask_count))

    def is_state_pos_flat(self, state):
        if state[0] == 0.5:
            return True
        else:
            return False

    def is_state_pos_long(self, state):
        if state[0] == 1:
            return True
        else:
            return False

    def is_state_pos_short(self, state):
        if state[0] == 0:
            return True
        else:
            return False

    def is_state_at_max_qty(self, state):
        if state[0] != 0.5 and state[1] == 1:
            return True
        else:
            return False

    def get_valid_vector_long_only(self):
        valid_mask = np.array([1, 1, 1, 1])

        assert(not self.is_current_pos_short())

        # len(self.episode_rewards)
        if self.is_current_pos_flat():
            valid_mask = np.array([0, 1, 0, 1])
        elif self.is_current_pos_long() and self.get_current_pos_qty() == self.position_max_qty:
            valid_mask = np.array([1, 0, 1, 1])

        return valid_mask

    def get_valid_vector_long_short_nothing(self):
        valid_mask = np.array([1, 1, 1, 1])

        # len(self.episode_rewards)
        if self.is_current_pos_flat():
            valid_mask = np.array([1, 1, 0, 1])
        elif self.is_current_pos_long() and self.get_current_pos_qty() == self.position_max_qty:
            valid_mask = np.array([1, 0, 1, 1])
        elif self.is_current_pos_short() and self.get_current_pos_qty() == self.position_max_qty:
            valid_mask = np.array([0, 1, 1, 1])

        return valid_mask

    def get_valid_vector_long_short(self):
        valid_mask = np.array([1, 1, 1, 1])

        # len(self.episode_rewards)
        if self.is_current_pos_flat():
            valid_mask = np.array([1, 1, 0, 0])
        elif self.is_current_pos_long() and self.get_current_pos_qty() == self.position_max_qty:
            valid_mask = np.array([1, 0, 1, 1])
        elif self.is_current_pos_short() and self.get_current_pos_qty() == self.position_max_qty:
            valid_mask = np.array([0, 1, 1, 1])

        return valid_mask

    def get_valid_vector_mask(self):
        # POSSIBLE ACTIONS
        # 0 => short
        # 1 => long
        # 2 => close position
        # 3 => do nothing or next step

        valid_mask = self.get_valid_vector_long_only()
        # valid_mask = self.get_valid_vector_long_short_nothing()
        # valid_mask = self.get_valid_vector_long_short()

        return valid_mask

    def get_valid_vector_mask_probs(self):

        ret = self.get_valid_vector_mask_probs_long_only()
        # ret = self.get_valid_vector_mask_probs_long_short()

        return ret

    def get_valid_mask(self):
        valid_mask = self.get_valid_vector_mask()
        valid_mask = valid_mask.astype(bool)
        valid_mask = np.invert(valid_mask)
        valid_mask = torch.from_numpy(valid_mask).to(self.device)

        return valid_mask

    def get_valid_vector_mask_probs_long_only(self):
        valid_mask = self.get_valid_vector_mask()
        tot = np.sum(valid_mask)
        ret = []
        assert (tot > 0)

        if valid_mask[1] > 0:
            prob_long = 0.99
            probs = (1 - prob_long) / (tot - 1)
            valid_mask = valid_mask * probs
            valid_mask[1] = prob_long
            ret = valid_mask
        elif self.is_current_pos_long() and valid_mask[1] == 0 and valid_mask[3] == 1:
            prob_do_nothing = 0.99
            probs = (1 - prob_do_nothing) / (tot - 1)
            valid_mask = valid_mask * probs
            valid_mask[3] = prob_do_nothing
            ret = valid_mask
        else:
            ret = valid_mask / tot

        return ret

    def get_valid_vector_mask_probs_long_short(self):
        valid_mask = self.get_valid_vector_mask()
        tot = np.sum(valid_mask)
        ret = []
        assert (tot > 0)

        if self.is_current_pos_long() and valid_mask[1] == 1:
            prob_long = 0.99
            probs = (1 - prob_long) / (tot - 1)
            valid_mask = valid_mask * probs
            valid_mask[1] = prob_long
            ret = valid_mask
        elif self.is_current_pos_short() and valid_mask[0] == 1:
            prob_short = 0.99
            probs = (1 - prob_short) / (tot - 1)
            valid_mask = valid_mask * probs
            valid_mask[0] = prob_short
            ret = valid_mask
        elif (self.is_current_pos_long() and valid_mask[1] == 0 and valid_mask[3] == 1) or \
             (self.is_current_pos_short() and valid_mask[0] == 0 and valid_mask[3] == 1):
            prob_do_nothing = 0.99
            probs = (1 - prob_do_nothing) / (tot - 1)
            valid_mask = valid_mask * probs
            valid_mask[3] = prob_do_nothing
            ret = valid_mask
        else:
            ret = valid_mask / tot

        return ret

    # def get_valid_boolean_mask(self, next_states=None):
    #     # POSSIBLE ACTIONS
    #     # 0 => short
    #     # 1 => long
    #     # 2 => close position
    #     # 3 => do nothing or next step
    #
    #     if next_states is None:
    #         valid_mask = torch.tensor([False, False, False, False]).to(self.device)
    #         if self.is_current_pos_flat():
    #             valid_mask = torch.tensor([False, False, True, True]).to(self.device)
    #         elif self.is_current_pos_long() and self.get_current_pos_qty() == self.position_max_qty:
    #             valid_mask = torch.tensor([False, True, False, False]).to(self.device)
    #         elif self.is_current_pos_short() and self.get_current_pos_qty() == self.position_max_qty:
    #             valid_mask = torch.tensor([True, False, False, False]).to(self.device)
    #
    #         return valid_mask
    #     else:
    #         # STATE:
    #         # 0. pos -> 0 0.5 1
    #         # 1. qty -> [self.get_current_pos_qty() / self.position_max_qty]
    #         # 2. wall
    #         # 3. ticks
    #         # 4. features
    #         # valid_mask_tensor = None
    #         # batch_size = next_states.shape[0]
    #         #
    #         # for i in range(batch_size):
    #         #     valid_mask = torch.tensor([[False, False, False, False]], device=self.device)
    #         #
    #         #     if self.is_state_pos_flat(next_states[i]):
    #         #         valid_mask = torch.tensor([[False, False, True, True]], device=self.device)
    #         #     elif self.is_state_pos_long(next_states[i]) and self.is_state_at_max_qty(next_states[i]):
    #         #         valid_mask = torch.tensor([[False, True, False, False]], device=self.device)
    #         #     elif self.is_state_pos_short(next_states[i]) and self.is_state_at_max_qty(next_states[i]):
    #         #         valid_mask = torch.tensor([[True, False, False, False]], device=self.device)
    #         #
    #         #     if valid_mask_tensor is None:
    #         #         valid_mask_tensor = valid_mask
    #         #     else:
    #         #         valid_mask_tensor = torch.cat((valid_mask_tensor, valid_mask), dim=0)
    #
    #         valid_mask_tensor = np.empty(shape=(0, 4), dtype=bool)
    #         batch_size = next_states.shape[0]
    #
    #         for i in range(batch_size):
    #             valid_mask = np.array([False, False, False, False])
    #
    #             if self.is_state_pos_flat(next_states[i]):
    #                 valid_mask = np.array([False, False, True, True])
    #             elif self.is_state_pos_long(next_states[i]) and self.is_state_at_max_qty(next_states[i]):
    #                 valid_mask = np.array([False, True, False, False])
    #             elif self.is_state_pos_short(next_states[i]) and self.is_state_at_max_qty(next_states[i]):
    #                 valid_mask = np.array([True, False, False, False])
    #
    #             valid_mask_tensor = np.vstack((valid_mask_tensor, valid_mask))
    #
    #         return torch.from_numpy(valid_mask_tensor).to(device=self.device)

    # def is_action_valid(self, action):
    #     mask = self.get_valid_binary_mask()
    #     if mask[action].detach().cpu().numpy() == 1:
    #         return True
    #     else:
    #         return False

    def set_and_save_scaler(self, featureList, scaler_name):
        feature_info, price_info, scaler = norm_features(features=featureList, n_label_cols=self.num_feature_headers)
        pickle.dump(scaler, open(self.output_path + scaler_name, 'wb'))
        return feature_info, price_info, scaler

    def set_scaler_and_transform_features(self, scaler_name, clip_after_transform=True):
        pickle_file = open(self.output_path + scaler_name, 'rb')
        self.scaler = pickle.load(pickle_file)
        pickle_file.close()
        feature_info, price_info, scaler = norm_features(features=self.featureList,
                                                         n_label_cols=self.num_feature_headers,
                                                         test_scaler=self.scaler)
        if clip_after_transform:
            feature_info = np.clip(feature_info, a_min=0, a_max=1)
        self.feature_info = feature_info
        self.price_info = price_info

    def load_scaler(self, path):
        self.scaler = pickle.load(open(path, 'rb'))

    def sample(self):
        return self.np_random.randint(self.num_actions)

    def get_features_start_end_dates(self):
        start = str(int(self.featureList[0, 4]))
        end = str(int(self.featureList[-1, 4]))
        start_date = datetime.strptime(start, '%Y%m%d%H%M%S')
        end_date = datetime.strptime(end, '%Y%m%d%H%M%S')
        return "Start Date: {}, End Date:{}".format(start_date, end_date)

    def load_feature_list(self, file_path=None):
        if file_path.find('.zip') >= 0 or file_path.find('.json') >= 0:
            is_zip = file_path.find('.zip') >= 0
            self.featureList = load_json(file_path, is_zip)
        elif file_path.find('.npz') >= 0:
            self.featureList = load_npz(file_path)

    def load_episode_rewards(self):
        if len(self.episode_rewards) == 0:
            try:
                self.episode_rewards = pickle.load(open(self.data_file_path + '1ep_rewards.pickle', 'rb'))
            except:
                print('Episode rewards file not found...')
                logging.debug('Episode rewards file not found...')

    def save_episode_rewards(self):
        with open(self.output_path + '1ep_rewards.pickle', 'wb') as handle:
            pickle.dump(self.episode_rewards, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def randomize_feature_start(self, rand_train_range_start, rand_train_range_end):
        if rand_train_range_end > 0 and self.featureList is not None:
            start = randint(rand_train_range_start, rand_train_range_end)
            self.start_time_step = start
            self.current_time_step = start

    def get_max_timesteps(self):
        if self.max_time_steps == 0:
            return self.feature_info.shape[0] - 1
        else:
            return self.max_time_steps - 1

    def get_num_inputs(self):
        return self.feature_info.shape[1] + 2

    def reset(self):
        self.done = False
        self.current_time_step = self.start_time_step
        self.balance = 0
        self.reward_total = 0
        if self.shuffle_per_tran:
            self.current_time_step = self.get_shuffle_per_tran_current_time_step()
        self.shuffle_per_tran_tt_counter = 0
        self.transactions = np.empty(shape=(0, self.transaction_dim))
        self.executions = np.empty(shape=(0, 7))
        self.order_queue = np.empty(shape=(0, 6))
        self.reward_steps = np.array([[0, 0]])
        self.last_session_printed = False
        self.session_count = 0
        self.num_print_trans = 0
        self.max_accum_reward = 0
        self.max_accum_reward_low_point = 0
        self.max_accum_reward_start = 0
        self.max_accum_reward_low_point_end = 0
        self.max_accum_losses = np.array([[0, 0, 0, 0]])
        self.state = self.get_state()
        return self.state

    def get_current_price(self, time_step=-1):
        if time_step == -1:
            time_step = self.current_time_step
        return self.price_info[time_step][0]

    def get_current_bid(self, time_step=-1):
        if time_step == -1:
            time_step = self.current_time_step
        return self.price_info[time_step][1]

    def get_current_ask(self, time_step=-1):
        if time_step == -1:
            time_step = self.current_time_step
        return self.price_info[time_step][2]

    def is_current_pos_long(self):
        return self.get_current_pos_type() == POSTYPE_LONG

    def is_current_pos_short(self):
        return self.get_current_pos_type() == POSTYPE_SHORT

    def is_current_pos_flat(self):
        return self.get_current_pos_type() == POSTYPE_FLAT

    def get_current_pos_type(self):
        last_pos = POSTYPE_FLAT
        if self.executions.shape[0] > 0:
            last_pos = self.executions[-1, 4]
        return last_pos

    def get_current_pos_qty(self):
        last_pos_qty = 0
        if self.executions.shape[0] > 0:
            last_pos_qty = self.executions[-1, 5]
        return last_pos_qty

    def get_current_pos_price(self):
        last_pos_price = 0
        if self.executions.shape[0] > 0:
            last_pos_price = self.executions[-1, 6]
        return last_pos_price

    def get_current_pos_ts(self):
        last_time_step = 0
        if self.executions.shape[0] > 0:
            last_time_step = self.executions[-1, 0]
        return last_time_step

    def is_margin_call(self):
        if self.min_margin is not None:
            return self.get_realized_balance() + self.get_gross_unrealized_value() - self.min_margin <= 0
        else:
            return False

    def get_state(self, wall=0):

        # *********** MODIFY self.num_extra_features = 1
        # POSITION
        # First element of the state needs to be the position state.
        # This can be useful to filter valid actions
        # 0 -> LONG
        # 0.5 -> SHORT
        # 1 -> FLAT
        position = 0.5  # FLAT
        if self.is_current_pos_long():
            position = 1
        elif self.is_current_pos_short():
            position = 0
        state = [position / 1]  # Position Time Steps

        # POSITION QTY
        state = np.append(state, [self.get_current_pos_qty() / self.position_max_qty])

        # If we hit a wall 1
        # else it is 0
        state = np.append(state, [wall])

        # State for when we are at cutoff. Network should know that there are no possible transactions
        # when session is closed
        # senssion_ended = 0
        # if self.featureList[self.current_time_step][3] <= 1:
        #     senssion_ended = 1
        # state = np.append(state, [senssion_ended])

        # ADD SENSE OF EPISODE CULMINATION FEATURE. THIS CAN BE TIME STEPS FINISHING UP OR BALANCE
        # FINISHING UP. SINCE BALANCE IS MORE TRANSFERABLE TO REAL TIME, THEN LET'S USE THAT.
        # TRY TO USE ENOUGH $ TO GET THROUGH THE COMPLETE TIME STEP SERIES ON FIRST GO.
        # state = np.append(state, [(self.max_time_steps - self.current_time_step) / self.max_time_steps])
        # REALIZED
        # realized = self.get_realized_balance()
        # state = np.append(state, [realized / self.init_balance])  # Unrealized

        # UNREALIZED
        # unr = (self.get_exit_unrlz_value() / self.tick_value) / 500
        # state = np.append(state, [unr])  # self.init_balance])  # Unrealized
        # Get Ticks away instea of unrlz when using multiple QTY
        ticks = self.get_exit_ticks_away()
        min_val = -500
        max_val = 500
        ticks = (ticks - min_val) / (max_val - min_val)
        state = np.append(state, [ticks])  # self.init_balance])  # Unrealized

        # LAST POSITION RETURN. This can help the algorithm not enter same consecutive
        # positions, especially if last position had a negative return.
        # last_pos, last_return = self.get_last_transaction()
        # state = np.append(state, [last_pos])  # Last Position
        # state = np.append(state, [last_return])  # Last Return. This is always normalized

        # STEPS. This could be useful in case we want to reward or penalize the
        # amount of time steps in position or inactivity. Position time steps should
        # be regularized by gamma. The shorter gamma is the more short term the policy
        # should be.
        # state = np.append(state, [self.inactive_time_step / 5000])  # Inactive Time Steps
        # state = np.append(state, [self.position_time_step / 5000])  # Position Time Steps

        # ENTER ACTION
        # l_votes, s_votes = self.enter_action_votes()
        # state = np.append(state, [l_votes])  # Inactive Time Steps

        # Add regular FEATURES
        self.state = np.append(state, self.feature_info[self.current_time_step])

        self.previous_state = self.state

        return self.state

    def get_max_accum_loss(self):
        max_accum_loss = self.max_accum_losses[np.argmax(self.max_accum_losses[:, 0] - self.max_accum_losses[:, 2])]
        return max_accum_loss[0] - max_accum_loss[2], max_accum_loss[1], max_accum_loss[3]

    def submit_order(self, action=None, qty=1):
        # POSSIBLE ACTIONS
        # 0 => short
        # 1 => long
        # 2 => close position
        # 3 => do nothing or next step
        if action > 2:
            return

        # ORDER QUEUE
        # - Order Action
        # - Order Type
        # - Order Price
        # - Order Qty
        # - In Time Step
        # - Is Live Until Canceled (1 / 0). If 0 then order_queue_ts_alive

        # ORDER ACTIONS
        # EnterLong() generates OrderAction.Buy
        # ExitLong() generates OrderAction.Sell
        # EnterShort() generates OrderAction.SellShort
        # ExitShort() generates OrderAction.BuyToCover
        # ORDER_ACTION_BUY = 1
        # ORDER_ACTION_SELL = 2
        # ORDER_ACTION_SELL_SHORT = 3
        # ORDER_ACTION_BUY_TO_COVER = 4

        # ORDER TYPES
        # OrderType.Limit
        # OrderType.Market
        # OrderType.MIT
        # OrderType.StopMarket
        # OrderType.StopLimit
        # ORDER_TYPE_LIMIT = 1
        # ORDER_TYPE_MARKET = 2
        # ORDER_TYPE_STOP_MARKET = 3
        # ORDER_TYPE_STOP_LIMIT = 4

        order_action = 0
        order_type = 0
        order_price = 0
        order_qty = 0
        order_ts = self.current_time_step
        order_luc = 0

        if self.is_current_pos_flat():
            if action == 0:
                order_action = ORDER_ACTION_SELL_SHORT
                order_type = ORDER_TYPE_MARKET
                order_price = self.get_current_bid()
                order_qty = qty
                order_ts = self.current_time_step
                order_luc = 0
            elif action == 1:
                order_action = ORDER_ACTION_BUY
                order_type = ORDER_TYPE_MARKET
                order_price = self.get_current_ask()
                order_qty = qty
                order_ts = self.current_time_step
                order_luc = 0
        elif self.is_current_pos_long():
            if action == 0:
                order_action = ORDER_ACTION_SELL
                order_type = ORDER_TYPE_MARKET
                order_price = self.get_current_bid()
                order_qty = qty
                order_ts = self.current_time_step
                order_luc = 0
            elif action == 1:
                order_action = ORDER_ACTION_BUY
                order_type = ORDER_TYPE_MARKET
                order_price = self.get_current_ask()
                order_qty = qty
                order_ts = self.current_time_step
                order_luc = 0
            if action == 2:
                order_action = ORDER_ACTION_SELL
                order_type = ORDER_TYPE_MARKET
                order_price = self.get_current_bid()
                order_qty = self.get_current_pos_qty()
                order_ts = self.current_time_step
                order_luc = 0
        elif self.is_current_pos_short():
            if action == 0:
                order_action = ORDER_ACTION_SELL_SHORT
                order_type = ORDER_TYPE_MARKET
                order_price = self.get_current_bid()
                order_qty = qty
                order_ts = self.current_time_step
                order_luc = 0
            elif action == 1:
                order_action = ORDER_ACTION_BUY_TO_COVER
                order_type = ORDER_TYPE_MARKET
                order_price = self.get_current_ask()
                order_qty = qty
                order_ts = self.current_time_step
                order_luc = 0
            if action == 2:
                order_action = ORDER_ACTION_BUY_TO_COVER
                order_type = ORDER_TYPE_MARKET
                order_price = self.get_current_ask()
                order_qty = self.get_current_pos_qty()
                order_ts = self.current_time_step
                order_luc = 0

        order = np.array([order_action, order_type, order_price, order_qty, order_ts, order_luc])
        self.order_queue = np.vstack([self.order_queue, order])

    def process_order_queue(self):
        # run through each orders.
        # If it is eligible for execution add an execution

        for i in range(self.order_queue.shape[0]):
            order_action = self.order_queue[i, 0]
            order_type = self.order_queue[i, 1]
            order_price = self.order_queue[i, 2]
            order_qty = self.order_queue[i, 3]
            order_ts = self.order_queue[i, 4]
            order_luc = self.order_queue[i, 5]
            if order_type == ORDER_TYPE_MARKET:
                # ADD THE EXECUTION
                self.add_execution(self.current_time_step, order_action, order_qty, order_price)

                # REMOVE ORDER FROM QUEUE
                self.order_queue = np.delete(self.order_queue, i, 0)

    def get_step_reward(self):
        return self.reward_steps[-1, 0] - self.reward_steps[-2, 0] + self.reward_steps[-1, 1] - self.reward_steps[-2, 1]

    def get_realized_balance(self):
        return self.balance

    def get_exit_unrlz_value(self, time_step=-1):
        if time_step == -1:
            time_step = self.current_time_step
        if self.get_current_pos_type() == POSTYPE_LONG:
            exit_unrlz_value = (self.get_current_bid(time_step) - self.get_current_pos_price()) \
                               * self.get_current_pos_qty() * self.point_value - self.commission_per_order * self.get_current_pos_qty()
        elif self.get_current_pos_type() == POSTYPE_SHORT:
            exit_unrlz_value = (self.get_current_pos_price() - self.get_current_ask(time_step)) \
                               * self.get_current_pos_qty() * self.point_value - self.commission_per_order * self.get_current_pos_qty()
        else:
            exit_unrlz_value = 0

        return exit_unrlz_value

    def get_gross_unrealized_value(self, time_step=-1):
        if time_step == -1:
            time_step = self.current_time_step
        if self.get_current_pos_type() == POSTYPE_LONG:
            unrlz_value = (self.get_current_bid(
                time_step) - self.get_current_pos_price()) * self.get_current_pos_qty() * self.point_value
        elif self.get_current_pos_type() == POSTYPE_SHORT:
            unrlz_value = (self.get_current_pos_price() - self.get_current_ask(
                time_step)) * self.get_current_pos_qty() * self.point_value
        else:
            unrlz_value = 0

        return unrlz_value

    def get_exit_ticks_away(self, time_step=-1):
        if time_step == -1:
            time_step = self.current_time_step
        if self.get_current_pos_type() == POSTYPE_LONG:
            ticks_away = (self.get_current_bid(time_step) - self.get_current_pos_price()) / self.tick_size
        elif self.get_current_pos_type() == POSTYPE_SHORT:
            ticks_away = (self.get_current_pos_price() - self.get_current_ask(time_step)) / self.tick_size
        else:
            ticks_away = 0

        return ticks_away

    def exit_pos(self):
        self.submit_order(action=2, qty=1)
        self.process_order_queue()

    def add_execution(self, order_time_step, order_action, order_qty, order_price):
        # EnterLong() generates OrderAction.Buy
        # ExitLong() generates OrderAction.Sell
        # EnterShort() generates OrderAction.SellShort
        # ExitShort() generates OrderAction.BuyToCover

        # Executions
        # - Time Step
        # - Action (Buy / Sell)
        # - Quantity
        # - Fill Price
        # - Position Type (Long/Short)
        # - Position Qty
        # - Position Avg Price

        order_action_type = -1
        if order_action == ORDER_ACTION_BUY_TO_COVER or order_action == ORDER_ACTION_BUY:
            order_action_type = POSTYPE_LONG
        elif order_action == ORDER_ACTION_SELL or order_action == ORDER_ACTION_SELL_SHORT:
            order_action_type = POSTYPE_SHORT
        else:
            AssertionError()

        pos_ts = order_time_step
        pos_qty = order_qty
        pos_avg_price = order_price
        pos_type = order_action_type

        # ADJUST ORDER COMMISSIONS TO BALANCE
        self.balance -= order_qty * self.commission_per_order

        if self.executions.shape[0] > 0:
            # Get Last Position info
            last_pos_time_step = self.get_current_pos_ts()
            last_pos = self.get_current_pos_type()
            last_pos_qty = self.get_current_pos_qty()
            last_pos_avg_price = self.get_current_pos_price()

            pos_ts = last_pos_time_step

            if last_pos == POSTYPE_FLAT:
                pos_type = order_action_type
                pos_qty = order_qty
                pos_avg_price = order_price
                pos_ts = order_time_step
            elif order_action_type == POSTYPE_LONG and last_pos == POSTYPE_LONG:
                pos_type = POSTYPE_LONG
                pos_qty = last_pos_qty + order_qty
                pos_avg_price = ((last_pos_avg_price * last_pos_qty) + (order_price * order_qty)) / (
                        last_pos_qty + order_qty)
            elif order_action_type == POSTYPE_SHORT and last_pos == POSTYPE_SHORT:
                pos_type = POSTYPE_SHORT
                pos_qty = last_pos_qty + order_qty
                pos_avg_price = ((last_pos_avg_price * last_pos_qty) + (order_price * order_qty)) / (
                        last_pos_qty + order_qty)
            elif order_action_type == POSTYPE_SHORT and last_pos == POSTYPE_LONG:
                if last_pos_qty > order_qty:
                    # DEFINE REMAINING POSITION
                    pos_type = POSTYPE_LONG
                    pos_qty = last_pos_qty - order_qty
                    pos_avg_price = last_pos_avg_price

                    # ADD A TRANSACTION FOR MATCHING QTY
                    self.add_transaction(POSTYPE_LONG, order_qty, last_pos_time_step, last_pos_avg_price,
                                         order_time_step, order_price)

                    # ADJUST BALANCE
                    self.balance += (order_price - last_pos_avg_price) * order_qty * self.point_value
                elif last_pos_qty < order_qty:
                    # DEFINE REMAINING POSITION
                    pos_type = POSTYPE_SHORT
                    pos_qty = order_qty - last_pos_qty
                    pos_avg_price = order_price

                    # ADD A TRANSACTION FOR MATCHING QTY
                    self.add_transaction(POSTYPE_LONG, last_pos_qty, last_pos_time_step, last_pos_avg_price,
                                         order_time_step, order_price)

                    # ADJUST BALANCE
                    self.balance += (order_price - last_pos_avg_price) * last_pos_qty * self.point_value
                else:
                    # DEFINE REMAINING POSITION
                    pos_type = POSTYPE_FLAT
                    pos_qty = 0
                    pos_avg_price = None

                    # ADD A TRANSACTION FOR MATCHING QTY
                    self.add_transaction(POSTYPE_LONG, order_qty, last_pos_time_step, last_pos_avg_price,
                                         order_time_step, order_price, True)

                    # ADJUST BALANCE
                    self.balance += (order_price - last_pos_avg_price) * order_qty * self.point_value

                    # RESET POSITION_TIME_STEP AND INACTIVE TIME_STEP
                    self.inactive_time_step = 0
                    self.position_time_step = 0
            elif order_action_type == POSTYPE_LONG and last_pos == POSTYPE_SHORT:
                if last_pos_qty > order_qty:
                    # DEFINE REMAINING POSITION
                    pos_type = POSTYPE_SHORT
                    pos_qty = last_pos_qty - order_qty
                    pos_avg_price = last_pos_avg_price

                    # ADD A TRANSACTION FOR MATCHING QTY
                    self.add_transaction(POSTYPE_SHORT, order_qty, last_pos_time_step, last_pos_avg_price,
                                         order_time_step, order_price)

                    # ADJUST BALANCE
                    self.balance += (last_pos_avg_price - order_price) * order_qty * self.point_value
                elif last_pos_qty < order_qty:
                    # DEFINE REMAINING POSITION
                    pos_type = POSTYPE_LONG
                    pos_qty = order_qty - last_pos_qty
                    pos_avg_price = order_price

                    # ADD A TRANSACTION FOR MATCHING QTY
                    self.add_transaction(POSTYPE_SHORT, last_pos_qty, last_pos_time_step, last_pos_avg_price,
                                         order_time_step, order_price)
                    # ADJUST BALANCE
                    self.balance += (last_pos_avg_price - order_price) * last_pos_qty * self.point_value
                else:
                    # DEFINE REMAINING POSITION
                    pos_type = POSTYPE_FLAT
                    pos_qty = 0
                    pos_avg_price = None

                    # ADD A TRANSACTION FOR MATCHING QTY
                    self.add_transaction(POSTYPE_SHORT, order_qty, last_pos_time_step, last_pos_avg_price,
                                         order_time_step, order_price, True)
                    # ADJUST BALANCE
                    self.balance += (last_pos_avg_price - order_price) * order_qty * self.point_value

                    # RESET POSITION_TIME_STEP AND INACTIVE TIME_STEP
                    self.inactive_time_step = 0
                    self.position_time_step = 0

        execution = np.array(
            [pos_ts, order_action_type, order_qty, order_price, pos_type, pos_qty, pos_avg_price])
        self.executions = np.vstack([self.executions, execution])

    def add_transaction(self, pos_type, pos_qty, in_time_step, in_price, out_time_step, out_price, is_flat=False):
        # TRANSACTIONS:
        # 0 - pos_type
        # 1 - in_time_step
        # 2 - pos_qty
        # 3 - in_price
        # 4 - out_time_step
        # 5 - last_price - --> NOT USED
        # 6 - out_price
        # 7 - num_time_steps
        # 8 - pnl - --> NOT USED
        # 9 - net_pnl
        # 10 - ticks
        # 11 - commissions
        # 12 - drawdown
        commissions = 2 * pos_qty * self.commission_per_order
        num_time_steps = out_time_step - in_time_step
        net_pnl = 0
        if pos_type == POSTYPE_LONG:
            net_pnl = (out_price - in_price) * pos_qty * self.point_value - commissions
        elif pos_type == POSTYPE_SHORT:
            net_pnl = (in_price - out_price) * pos_qty * self.point_value - commissions
        else:
            AssertionError()
        ticks = (out_price - in_price) / self.tick_size
        drawdown = self.max_drawdown

        # Calculate max accumulated reward
        tot_pnl = self.get_tot_tran_pnl()
        if tot_pnl > self.max_accum_reward:
            self.max_accum_reward = tot_pnl
            self.max_accum_reward_low_point = tot_pnl
            self.max_accum_reward_start = in_time_step
            self.max_accum_reward_low_point_end = in_time_step
            max_acc = np.array([tot_pnl, in_time_step, tot_pnl, in_time_step])
            self.max_accum_losses = np.vstack([self.max_accum_losses, max_acc])
        else:
            if tot_pnl < self.max_accum_reward_low_point:
                self.max_accum_reward_low_point = tot_pnl
                self.max_accum_reward_low_point_end = out_time_step
                self.max_accum_losses[-1, 2] = tot_pnl
                self.max_accum_losses[-1, 3] = out_time_step

        tmp_transaction = np.array(
            [pos_type, in_time_step, pos_qty, in_price, out_time_step, 0, out_price, num_time_steps,
             0, net_pnl, ticks, commissions, drawdown])
        self.transactions = np.vstack([self.transactions, tmp_transaction])

        self.transaction_added = True

        # if shuffling per transaction we must trigger the reset of starting step.
        if self.shuffle_per_tran and is_flat:
            # eliminate this entry point from our list if it had a postive return
            # if net_pnl >= 0:
            self.shuffle_per_tran_step_list = list(set(self.shuffle_per_tran_step_list) - {int(in_time_step)})
            self.shuffle_per_tran_trigger = True
            if len(self.shuffle_per_tran_step_list) % 1000 == 0:
                print("{} entry points left.".format(len(self.shuffle_per_tran_step_list)))
                logging.debug("{} entry points left.".format(len(self.shuffle_per_tran_step_list)))

    def get_tot_tran_pnl(self):
        tot_pnl = 0
        if self.transactions.shape[0] > 0:
            tot_pnl = np.sum(self.transactions[:, 9])
        return float(tot_pnl)

    def get_shuffle_per_tran_current_time_step(self):
        if len(self.shuffle_per_tran_step_list) == 0:
            if self.end_time_step is not None:
                i_range = int(self.end_time_step * .96)
            else:
                i_range = int(self.featureList.shape[0] * .96)
            self.shuffle_per_tran_step_list = list(range(self.start_time_step, i_range))
            print("Generating new shuffle list")
            logging.debug("Generating new shuffle list")

        return choice(self.shuffle_per_tran_step_list)

    # def set_table_col_names(self):
    #     self.table.set_column_header("ts", col=0)
    #     self.table.set_column_header("current_ts", col=1)
    #     self.table.set_column_header("tot_trans", col=2)
    #     self.table.set_column_header("tot_pnl", col=3)
    #     self.table.set_column_header("pos", col=4)
    #     self.table.set_column_header("unr_val", col=5)
    #     self.table.set_column_header("pos_qty", col=6)
    #     self.table.set_column_header("max_num_tt", col=7)
    #     self.table.set_column_header("avg_long_win", col=8)
    #     self.table.set_column_header("avg_long_loss", col=9)
    #     self.table.set_column_header("avg_long_win_ts", col=10)
    #     self.table.set_column_header("avg_long_loss_ts", col=11)
    #     self.table.set_column_header("avg_short_win", col=12)
    #     self.table.set_column_header("avg_short_loss", col=13)
    #     self.table.set_column_header("avg_short_win_ts", col=14)
    #     self.table.set_column_header("avg_short_loss_ts", col=15)
    #     self.table.set_column_header("num_long, num_short", col=16)
    #     self.table.set_column_header("num_wins_long", col=17)
    #     self.table.set_column_header("num_losses_long", col=18)
    #     self.table.set_column_header("num_wins_short", col=19)
    #     self.table.set_column_header("num_losses_short", col=20)

    def print_transaction_summary2(self, session_expired):
        # TRANSACTIONS:
        # 0 - pos_type
        # 1 - in_time_step
        # 2 - pos_qty
        # 3 - in_price
        # 4 - out_time_step
        # 5 - last_price - --> NOT USED
        # 6 - out_price
        # 7 - num_time_steps
        # 8 - pnl - --> NOT USED
        # 9 - net_pnl
        # 10 - ticks
        # 11 - commissions
        # 12 - drawdown
        try:
            # print a new line when we start printing for the first time.
            if self.num_print_trans == 0:
                print("\n")
                logging.debug("\n")
            self.num_print_trans += 1

            if session_expired:
                self.session_count += 1
                self.last_session_printed = True
            # avg_num_tt = np.mean(self.transactions[:, 7])
            max_num_tt = np.max(self.transactions[:, 7])
            # avg_pnl = np.mean(self.transactions[:, 9])
            tot_pnl = np.sum(self.transactions[:, 9])

            num_long = self.transactions[np.where(self.transactions[:, 0] == POSTYPE_LONG)].shape[0]
            num_short = self.transactions[np.where(self.transactions[:, 0] == POSTYPE_SHORT)].shape[0]
            avg_long_win = 0
            avg_long_win_tt = 0
            avg_long_loss = 0
            avg_long_loss_tt = 0
            avg_short_win = 0
            avg_short_win_tt = 0
            avg_short_loss = 0
            avg_short_loss_tt = 0
            avg_qty_long = 0
            avg_qty_short = 0
            max_qty_long = 0
            max_qty_short = 0
            if num_long > 0:
                trans = self.transactions[:, 9][
                    np.where((self.transactions[:, 9] >= 0) & (self.transactions[:, 0] == POSTYPE_LONG))]
                if len(trans) > 0: avg_long_win = np.mean(trans)

                trans = self.transactions[:, 7][
                    np.where((self.transactions[:, 9] >= 0) & (self.transactions[:, 0] == POSTYPE_LONG))]
                if len(trans) > 0: avg_long_win_tt = np.mean(trans)

                trans = self.transactions[:, 9][
                    np.where((self.transactions[:, 9] < 0) & (self.transactions[:, 0] == POSTYPE_LONG))]
                if len(trans) > 0: avg_long_loss = np.mean(trans)

                trans = self.transactions[:, 7][
                    np.where((self.transactions[:, 9] < 0) & (self.transactions[:, 0] == POSTYPE_LONG))]
                if len(trans) > 0: avg_long_loss_tt = np.mean(trans)

                trans = self.transactions[:, 2][
                    np.where((self.transactions[:, 0] == POSTYPE_LONG))]
                if len(trans) > 0:
                    avg_qty_long = np.mean(trans)
                    max_qty_long = np.max(trans)

            if num_short > 0:
                trans = self.transactions[:, 9][
                    np.where((self.transactions[:, 9] >= 0) & (self.transactions[:, 0] == POSTYPE_SHORT))]
                if len(trans) > 0: avg_short_win = np.mean(trans)

                trans = self.transactions[:, 7][
                    np.where((self.transactions[:, 9] >= 0) & (self.transactions[:, 0] == POSTYPE_SHORT))]
                if len(trans) > 0: avg_short_win_tt = np.mean(trans)

                trans = self.transactions[:, 9][
                    np.where((self.transactions[:, 9] < 0) & (self.transactions[:, 0] == POSTYPE_SHORT))]
                if len(trans) > 0: avg_short_loss = np.mean(trans)

                trans = self.transactions[:, 7][
                    np.where((self.transactions[:, 9] < 0) & (self.transactions[:, 0] == POSTYPE_SHORT))]
                if len(trans) > 0: avg_short_loss_tt = np.mean(trans)

                trans = self.transactions[:, 2][
                    np.where((self.transactions[:, 0] == POSTYPE_SHORT))]
                if len(trans) > 0:
                    avg_qty_short = np.mean(trans)
                    max_qty_short = np.max(trans)

            time_step = self.transactions[-1][4]
            tot_trans = self.transactions.shape[0]
            num_wins_long = \
                np.array(np.where((self.transactions[:, 9] > 0) & (self.transactions[:, 0] == POSTYPE_LONG))).shape[
                    1]
            num_losses_long = \
                np.array(np.where((self.transactions[:, 9] < 0) & (self.transactions[:, 0] == POSTYPE_LONG))).shape[
                    1]
            num_wins_short = \
                np.array(
                    np.where((self.transactions[:, 9] > 0) & (self.transactions[:, 0] == POSTYPE_SHORT))).shape[1]
            num_losses_short = \
                np.array(
                    np.where((self.transactions[:, 9] < 0) & (self.transactions[:, 0] == POSTYPE_SHORT))).shape[1]
            # avg_drawdown = np.mean(self.transactions[:, 12])
            # max_drawdown = np.min(self.transactions[:, 12])
            unr_val = 0
            if not self.is_current_pos_flat():
                unr_val = self.get_gross_unrealized_value()

            pos = ""
            if self.is_current_pos_long(): pos = "L"
            if self.is_current_pos_short(): pos = "S"

            if self.shuffle_per_tran:
                txt = "TimeStep:{}/{}, TotTrans:{:>4}, TotPnl:{:6.2f}, Unrlzd:{}{:5.2f}, Qty:{}, Avg Qty L/S:{:4.0f}/{:4.0f}, MaxTS:{:>5}, " \
                      "L-Win/Loss:{:5.2f}/{:5.2f}, L-TS-W/L:{:5.2f}/{:5.2f}, "\
                      "S-Win/Loss:{:5.2f}/{:5.2f}, S-TS-W/L:{:5.2f}/{:5.2f}, "\
                      "L/S:{:>5}/{:>5}, "\
                      "W/L Long:{:>5}/{:>5}, W/L Short:{:>5}/{:>5}".format(time_step, self.shuffle_per_tran_tt_counter, tot_trans, tot_pnl, pos, unr_val,
                           self.get_current_pos_qty(),
                           avg_qty_long, avg_qty_short,
                           max_num_tt, avg_long_win, avg_long_loss, avg_long_win_tt, avg_long_loss_tt,
                           avg_short_win, avg_short_loss, avg_short_win_tt, avg_short_loss_tt,
                           num_long, num_short,
                           num_wins_long, num_losses_long, num_wins_short, num_losses_short)
                print(txt)
                logging.debug(txt)
            else:
                txt = "TimeStep:{:>7}/{}, TotTrans:{:>4}, TotPnl:{:6.2f}, Unrlzd:{}{:5.2f}, Qty:{}, Avg Qty L/S:{:4.0f}/{:4.0f}, MaxTS:{:>5}, "\
                    "L-Win/Loss:{:5.2f}/{:5.2f}, L-TS-W/L:{:5.2f}/{:5.2f}, "\
                    "S-Win/Loss:{:5.2f}/{:5.2f}, S-TS-W/L:{:5.2f}/{:5.2f}, "\
                    "L/S:{:>5}/{:>5}, "\
                    "W/L Long:{:>5}/{:>5}, W/L Short:{:>5}/{:>5}".format(time_step, self.current_time_step, tot_trans, tot_pnl, pos, unr_val,
                           self.get_current_pos_qty(),
                           avg_qty_long, avg_qty_short, max_num_tt,
                           avg_long_win, avg_long_loss, avg_long_win_tt, avg_long_loss_tt,
                           avg_short_win, avg_short_loss, avg_short_win_tt, avg_short_loss_tt,
                           num_long, num_short,
                           num_wins_long, num_losses_long, num_wins_short, num_losses_short)
                print(txt)
                logging.debug(txt)

            if session_expired:
                print('session {}, final time step {}...'.format(self.session_count, self.current_time_step))
                logging.debug('session {}, final time step {}...'.format(self.session_count, self.current_time_step))

            if self.max_time_steps - self.current_time_step <= 1:
                print('Last step...{}/{}'.format(self.max_time_steps, self.current_time_step))
                logging.debug('Last step...{}/{}'.format(self.max_time_steps, self.current_time_step))
        except:
            error = True

    def plot_episode_returns(self, show=True):
        # change if NUMPY
        # plt.plot(np.asnumpy(self.episode_rewards))
        plt.plot(self.episode_rewards)
        plt.axhline(y=0, color='silver', linestyle='-')
        plt.xlabel = 'Episode'
        plt.ylabel = 'PnL'
        try:
            plt.savefig(self.output_path + 'plot_episode_returns.png')
        except:
            print("An exception occurred while printing " + 'plot_episode_returns.png')
            logging.debug("An exception occurred while printing " + 'plot_episode_returns.png')
        if show: plt.show()
        plt.clf()

    def plot_prices(self, show=True):
        start = self.start_time_step
        end = self.end_time_step
        if self.end_time_step == None: end = self.featureList.shape[0]
        plt.plot(self.featureList[start:end, 0])
        plt.xlabel = 'Time Steps'
        plt.ylabel = 'Price'
        try:
            plt.savefig(self.output_path + 'plot_prices.png')
        except:
            print("An exception occurred while printing " + 'plot_prices.png')
            logging.debug("An exception occurred while printing " + 'plot_prices.png')

        if show: plt.show()
        plt.clf()

    def plot_price_and_returns(self, show=True):
        # TRANSACTIONS:
        # 0 - pos_type
        # 1 - in_time_step
        # 2 - last_price - --> NOT USED
        # 3 - in_price
        # 4 - out_time_step
        # 5 - last_price - --> NOT USED
        # 6 - out_price
        # 7 - num_time_steps
        # 8 - pnl - --> NOT USED
        # 9 - net_pnl
        # 10 - ticks
        # 11 - commissions
        # 12 - drawdown

        # Print Rewards
        # x = self.env.transactions[:, 1]
        y = self.transactions[:, 9].cumsum()
        # fig, (ax1, ax2) = plt.subplots(2)
        fig = plt.figure()
        gs = fig.add_gridspec(ncols=1, nrows=2, hspace=0)
        axs = gs.subplots(sharex=True, sharey=False)
        color = 'tab:green'
        # ax1.set_xlabel('Time Step')
        axs[0].set_ylabel('Reward', color=color)
        for i in range(self.transactions.shape[0]):
            x1 = self.transactions[i][1]
            x2 = self.transactions[i][4]
            y1 = 0
            if i > 0:
                y1 = y[i - 1]
            y2 = y[i]
            if self.transactions[i][0] == POSTYPE_LONG:
                axs[0].plot([x1, x2], [y1, y2], 'g')
            else:
                axs[0].plot([x1, x2], [y1, y2], 'r')
        # ax1.plot(x, y, color=color)
        axs[0].tick_params(axis='y', labelcolor=color)
        axs[0].axhline(y=0, color='silver', linestyle='-')

        # ax1.set_xlabel('Time Step')
        # ax1.xaxis.label.set_color('cyan')
        # ax1.tick_params(axis='x', colors='cyan')

        # Print Prices
        y2 = self.price_info[self.start_time_step:self.current_time_step, 0]
        x2 = []
        x_i = self.start_time_step
        for i in range(y2.shape[0]):
            x2.append(x_i)
            if i % 1 == 0:
                x_i += 1
        # ax2 = ax1.twinx()
        color = 'tab:blue'
        axs[1].set_xlabel("Time Step. Episode {}".format(len(self.episode_rewards)-1))
        axs[1].xaxis.label.set_color('cyan')
        axs[1].tick_params(axis='x', colors='cyan')
        axs[1].set_ylabel('Price', color=color)  # we already handled the x-label with ax1
        axs[1].plot(x2, y2, color='lightgray')

        for i in range(self.transactions.shape[0]):
            x1 = self.transactions[i][1]
            x2 = self.transactions[i][4]
            y1 = self.transactions[i][3]
            y2 = self.transactions[i][6]
            if self.transactions[i][0] == POSTYPE_LONG:
                axs[1].plot([x1, x2], [y1, y2], 'g')
            else:
                axs[1].plot([x1, x2], [y1, y2], 'r')

        axs[1].tick_params(axis='y', labelcolor=color)

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        fig.set_size_inches(11, 8)
        try:
            plt.savefig(self.output_path + 'plot_price_and_returns_' + str(len(self.episode_rewards)-1) + '.png')
        except:
            print("An exception occurred while printing " + 'plot_price_and_returns_' + str(len(self.episode_rewards)) + '.png')
            logging.debug("An exception occurred while printing " + 'plot_price_and_returns_' + str(len(self.episode_rewards)) + '.png')
        if show: plt.show()
        plt.clf()

    def plot_returns(self, show=True):
        # TRANSACTIONS:
        # 0 - pos_type
        # 1 - in_time_step
        # 2 - last_price - --> NOT USED
        # 3 - in_price
        # 4 - out_time_step
        # 5 - last_price - --> NOT USED
        # 6 - out_price
        # 7 - num_time_steps
        # 8 - pnl - --> NOT USED
        # 9 - net_pnl
        # 10 - ticks
        # 11 - commissions
        # 12 - drawdown

        # Print Rewards
        # x = self.env.transactions[:, 1]
        y = self.transactions[:, 9].cumsum()
        # fig, (ax1, ax2) = plt.subplots(2)
        fig = plt.figure()
        gs = fig.add_gridspec(ncols=1, nrows=2, hspace=0)
        axs = gs.subplots(sharex=True, sharey=False)
        color = 'tab:green'
        # ax1.set_xlabel('Time Step')
        axs[0].set_ylabel('Reward', color=color)
        for i in range(self.transactions.shape[0]):
            x1 = self.transactions[i][1]
            x2 = self.transactions[i][4]
            y1 = 0
            if i > 0:
                y1 = y[i - 1]
            y2 = y[i]
            if self.transactions[i][0] == POSTYPE_LONG:
                axs[0].plot([x1, x2], [y1, y2], 'g')
            else:
                axs[0].plot([x1, x2], [y1, y2], 'r')

        # ax1.plot(x, y, color=color)
        axs[0].tick_params(axis='y', labelcolor=color)

        axs[0].axhline(y=0, color='silver', linestyle='-')

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        fig.set_size_inches(11, 8)
        try:
            plt.savefig(self.output_path + 'plot_returns_' + str(len(self.episode_rewards)) + '.png')
        except:
            print("An exception occurred while printing " + 'plot_returns_' + str(len(self.episode_rewards)) + '.png')
            logging.debug("An exception occurred while printing " + 'plot_returns_' + str(len(self.episode_rewards)) + '.png')

        if show: plt.show()
        plt.clf()

    def step(self, action=3):

        # USE THIS INSTEAD OF ACTION MASKING
        # 0 -> short
        # 1 -> long
        # 2 -> close position
        # 3 -> do nothing
        # action = act
        # if self.is_current_pos_flat() and act >= 2:
        #     if act == 2:
        #         action = 0
        #     else:
        #         action = 1
        # elif self.is_current_pos_long() and self.get_current_pos_qty() == self.position_max_qty and act == 1:
        #     action = 3
        # elif self.is_current_pos_short() and self.get_current_pos_qty() == self.position_max_qty and act == 0:
        #     action = 3

        # if action == 0:
        #     a = 0
        # elif action == 1:
        #     a = 1
        # elif action == 2:
        #     a = 2
        # elif action == 3:
        #     a = 3

        minutes_left_in_session = self.featureList[self.current_time_step][3]
        session_expired = (minutes_left_in_session <= 1) or (self.max_time_steps - self.current_time_step <= 1)
        if not session_expired:
            self.last_session_printed = False

        # VALIDATE ORDER Validate order_action through order_action masking. Return same state with negative reward
        # if order_action was at fault.
        # if not self.is_action_valid(action):
        #     return self.get_state(wall=1), -self.mask_wall_value, False, 0

        if not session_expired:
            self.submit_order(action, qty=1)
            self.process_order_queue()

        # CHECK FOR MAX PROFIT OR MAX DRAW DOWN WAS HIT TO EXIT
        exit_target = self.take_profit
        exit_stop = self.take_loss
        exit_target_flag = (self.get_gross_unrealized_value() >= exit_target)
        exit_stop_flag = (self.get_gross_unrealized_value() <= -exit_stop)

        if self.is_current_pos_flat():
            self.max_exit_unrlz = 0
        self.max_exit_unrlz = max(self.max_exit_unrlz, self.get_exit_unrlz_value())
        exit_max_net_drawdown = (self.exit_net_drawdown is not None and
                                 (self.get_exit_unrlz_value() - self.max_exit_unrlz <= -self.exit_net_drawdown))

        exit_pos = exit_target_flag or exit_stop_flag or exit_max_net_drawdown

        # CHECK FOR END OF TIME STEPS TO EXIT POSITION
        if session_expired and (not self.is_current_pos_flat()):
            exit_pos = True
            print('Exiting none flat position on last time step...{}'.format(self.current_time_step))
            logging.debug('Exiting none flat position on last time step...{}'.format(self.current_time_step))

        # If transaction is still open and episode is done, we need to close the transaction
        if (self.done and not self.is_current_pos_flat()) or exit_pos:
            self.exit_pos()

        self.done = self.is_margin_call() or (self.max_time_steps - self.current_time_step <= 1) or \
                    (self.shuffle_per_tran_trigger and self.shuffle_per_tran_tt_done is not None and
                     self.shuffle_per_tran_tt_counter >= self.shuffle_per_tran_tt_done)

        # Adjust MAX Drawdown
        if not self.is_current_pos_flat():
            self.max_drawdown = min(self.max_drawdown, self.get_gross_unrealized_value())

        # Move inactive or position time step
        if self.is_current_pos_flat():
            self.inactive_time_step += 1
            self.position_time_step = 0
        else:
            self.inactive_time_step = 0
            self.position_time_step += 1

        # PRINT SUMMARY
        if (session_expired and not self.last_session_printed) or (
                (self.current_time_step % 500 == 0) and self.transactions.shape[0] > 0):
            self.print_transaction_summary2(session_expired)

        # ************************************************
        # FROM HERE BELOW WE BUILD THE REWARD + NEXT STATE
        # ************************************************

        # This holds all of the time step values
        self.reward_steps = np.vstack((self.reward_steps, [self.balance, self.get_gross_unrealized_value()]))

        reward = self.get_step_reward()

        ret_reward = reward + self.incentives
        # THIS WILL DOUBLE THE REWARD ON CLOSE
        # if self.transaction_added:
        #     tran_reward = self.transactions[-1, 9]
        #     ret_reward += tran_reward
        #     self.transaction_added = False
        self.reward_total += ret_reward

        # CAREFUL
        # Here below we move the time step and give the next state back
        # Important to make sure current time step is consistent with STATE from this point forward.
        # *** ALL CALCULATIONS BELOW WILL BE FOR TIME STEP + 1***

        # Move one time step
        self.current_time_step += 1

        # if shuffling per transaction we will check if we need to random select a current time step
        # we need to do this prior to getting current state
        if self.shuffle_per_tran_trigger:
            self.current_time_step = self.get_shuffle_per_tran_current_time_step()
            self.shuffle_per_tran_trigger = False
        self.shuffle_per_tran_tt_counter += 1

        # Retrieve current state for current time step
        state = self.get_state()

        # Info is set to zero
        info = 0

        if self.done:
            self.episode_rewards = np.append(self.episode_rewards, self.get_tot_tran_pnl())
            self.save_episode_rewards()
            # Safer if you plot from Agent after saving the model.
            # self.plot_episode_returns()
            # self.plot_price_and_returns()

        # print("Pos:{}, Qty:{}, Balance:{:5.2f}, Tot Rewards:{:5.2f} ".format(self.get_current_pos_type(), self.get_current_pos_qty(), self.balance, self.reward_total))

        return state, ret_reward, self.done, info

    def render(self, mode='human', close=False):
        # Renders the environment to the screen
        # This should replace the printing.
        print_nothing = True

    def test_conciliation(self):
        # LONG
        self.step(1)
        self.step(3)
        self.step(1)
        self.step(3)
        self.step(1)
        self.step(3)
        self.step(1)
        self.step(3)
        self.step(3)
        self.step(3)
        self.step(3)
        self.step(0)
        self.step(3)
        self.step(3)
        self.step(0)
        self.step(3)
        self.step(3)
        self.step(0)
        self.step(3)
        self.step(3)
        self.step(0)

        # LONG
        self.step(1)
        self.step(1)
        self.step(3)
        self.step(3)
        self.step(3)
        self.step(3)
        self.step(3)
        self.step(3)
        self.step(2)

        # SHORT
        self.step(0)
        self.step(3)
        self.step(0)
        self.step(3)
        self.step(0)
        self.step(3)
        self.step(0)
        self.step(3)
        self.step(3)
        self.step(3)
        self.step(3)
        self.step(1)
        self.step(3)
        self.step(3)
        self.step(1)
        self.step(3)
        self.step(3)
        self.step(1)
        self.step(3)
        self.step(3)
        self.step(1)

        # SHORT
        self.step(0)
        self.step(0)
        self.step(3)
        self.step(3)
        self.step(3)
        self.step(3)
        self.step(3)
        self.step(3)
        self.step(2)
