from SynthTradingEnv.SynthCommons import to_3D, load_json
import numpy as np
import pickle
import matplotlib.pyplot as plt
from enum import Enum
from random import randint
import gym
from gym import spaces
from gym.utils import seeding
from deprecated import deprecated


# THIS HAS 3 ACTIONS BUY/SELL/HOLD

class PriceType(Enum):
    Last = 1
    Bid = 2
    Ask = 3


class PosType(Enum):
    Long = 1
    Short = 2
    Flat = 3

class Book:
    def __init__(self, pos=PosType.Flat, price=0, qty=0, realized_val=0):
        self.pos = pos
        self.price = price
        self.qty = qty
        self.realized_val = realized_val
        self.previous_realized_val = realized_val


class TradingEnv(gym.Env):
    """Custom Trading Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 featureList=None,
                 time_steps=1,
                 start_time_step=0,
                 end_time_step=None,
                 init_balance=100000,
                 tick_size=0.005,
                 tick_value=25,
                 commission_per_order=2.31,
                 take_profit=1e10,
                 take_loss=1e10,
                 max_time_steps=0,
                 shuffle=False,
                 normalize_return=True,
                 ff_on_loss=False,
                 min_ff_loss=500,
                 train_side=None,
                 skip_steps=1,
                 skip_incentive=0,
                 same_entry_penalty=-1000,
                 reward_for_each_step=False,
                 data_file_path='C:/Users/jortu/Documents/1Syntheticks-AI/',
                 data_file_name='SI-09-20-FEATURELIST.json',
                 load_historic_rewards=False):
        super(TradingEnv, self).__init__()

        # Environment Name
        self.env = "Sythetick Trading Env 1.0"

        # this will return the difference gained between step i-1 and step i. That way
        # true returns are fed back to Agent on each step.
        self.reward_for_each_step = reward_for_each_step
        # This is the penalty received for entering same side as previous while previous being
        # a negative return.
        self.same_entry_penalty = same_entry_penalty
        # make returns between 0 - 1
        self.normalize_return = normalize_return
        # if we want to fast forward to a winning or min_ff_loss state
        self.ff_on_loss = ff_on_loss
        # used in conjunction with ff_on_loss
        self.min_ff_loss = min_ff_loss
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

        # Shuffle the starting point
        self.shuffle = shuffle

        # This is the initial money balance with we start
        self.init_balance = init_balance
        # This is used in normalization of featureList. It will create a "Window" of n time_steps for each
        # row. This is inherited from LSTM, but not really used here.
        self.time_steps = time_steps
        # Starting point of the series
        self.start_time_step = start_time_step
        self.end_time_step = end_time_step
        # Keeps track of the current Time Step we are in at any moment
        self.current_time_step = start_time_step
        # The min amount that a futures instuments moves up and down. Example 0.005 in SI (silver) futures
        self.tick_size = tick_size
        # What 1 tick is valued at in money. Example $25.00 for 1 tick in SI (silver) futures
        self.tick_value = tick_value
        # Value of the commission charged per order per side (enter or exit) by the futures broker
        self.commission_per_order = commission_per_order
        # Max amount of orders that can be placed in one direction at a time. For now we are only using 1.
        self.max_per_side_qty = 1
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
        self.min_margin = 15000
        # This hold previous state. Not currently being used though.
        self.previous_state = []
        # Transactions is where we store all of the transactions (enter/exit pairs)
        self.transaction_dim = 13
        self.transactions = np.empty(shape=(0, self.transaction_dim))
        # Executions
        # - Time Step
        # - Action (Buy / Sell)
        # - Quantity
        # - Fill Price
        # - Position Type (Long/Short)
        # - Position Qty
        # - Position Avg Price
        self.executions = np.empty(shape=(0, 7))
        self.execution_order_reward = 0
        # This will hold temp transaction state while waiting for transaction to be closed
        self.tmp_transaction = np.empty(shape=(0, self.transaction_dim))
        # Book contains position:
        # If you keep stacking a position you need to get average
        # If you settle part or all of the position you need to
        # update the realized position. Also update realized every
        # time you buy/sell since you incur in commission.
        # BOOK: L/S, price, qty, realized value
        self.book = Book(realized_val=self.init_balance)
        # This contains all of the feature data. First columns include price (last, bid, ask) and
        # how many seconds left to terminate trading session. This data will be extracted form featureList.
        # The clean normalized features are included in feature_info
        self.featureList = featureList
        # This holds the maximum drawdown (loss or potential loss) throughout the episode
        self.max_drawdown = 0
        self.max_accum_reward = 0
        self.max_accum_reward_low_point = 0
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

        # Fast Forward mode used when forwarding trough min loss states. This can be used when we don't
        # want to exit a position until we have hit a min threshold of $XXX amount.
        self.ff = False
        self.min_ff_loss = 500

        # If we want to normalize returns sent back to (-1,+1), then we need to divide by a # that is the MAX expected
        # return of a single transactions. Don't expect clipping to -1, 1 would produce the same effect.
        self.max_return_norm = 20000

        # Used to calculate discounted value return per transaction
        self.tran_unrlz_value = []
        self.prev_unrlz_value = 0

        self.episode_rewards = []

        # Number of times transactions have been printed out.
        self.num_print_trans = 0

        if load_historic_rewards:
            self.load_episode_rewards()

        ### FROM HERE BELOW BEING ADAPTED FOR GYM
        self.np_random = None

        # Load Feature List
        self.load_feature_list(file_path=self.data_file_path_name)
        if self.end_time_step is not None and self.end_time_step > self.start_time_step:
            self.featureList = self.featureList[0:self.end_time_step]

        self.num_featureList_features = self.featureList.shape[1]
        self.num_feature_headers = 3
        self.num_extra_features = 1
        self.num_features = self.num_featureList_features + self.num_extra_features - self.num_feature_headers

        # 0 -> short
        # 1 -> long
        # 2 -> do nothing
        self.num_actions = 2

        # ****** Gym specific ******
        self.name = "Synthetick Trading Environment"
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

        ### FROM HERE UP BEING ADAPTED FOR GYM

        if self.featureList is not None and self.featureList.shape[1]:
            self.feature_info, self.price_info, self.scaler = self.set_and_save_scaler(self.featureList,
                                                                                       'scaler.pickle')
            self.max_time_steps = self.get_max_timesteps()

    def set_and_save_scaler(self, featureList, scaler_name):
        feature_info, price_info, scaler = to_3D(featureList_=featureList, timesteps=1)
        pickle.dump(self.scaler, open(scaler_name, 'wb'))
        return feature_info, price_info, scaler

    def load_scaler(self, path):
        self.scaler = pickle.load(open(path, 'rb'))

    def sample(self):
        return self.np_random.randint(self.num_actions)

    def load_feature_list(self, file_path=None):
        is_zip = file_path.find('.zip') >= 0
        self.featureList = load_json(file_path, is_zip)

    def load_episode_rewards(self):
        if len(self.episode_rewards) == 0:
            try:
                self.episode_rewards = pickle.load(open(self.data_file_path + '1ep_rewards.pickle', 'rb'))
            except:
                print('Episode rewards file not found...')

    def save_episode_rewards(self):
        with open(self.data_file_path + '1ep_rewards.pickle', 'wb') as handle:
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
        self.current_time_step = self.start_time_step
        if self.shuffle:
            start = 0
            end = int(self.feature_info.shape[0] * 0.75)
            self.current_time_step = randint(start, end)
        self.book = Book(realized_val=self.init_balance)
        self.tmp_transaction = np.empty(shape=(0, self.transaction_dim))
        self.transactions = np.empty(shape=(0, self.transaction_dim))
        self.executions = np.empty(shape=(0, 7))
        self.last_session_printed = False
        self.session_count = 0
        self.num_print_trans = 0
        self.max_accum_reward = 0
        self.max_accum_reward_low_point = 0
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

    def get_current_pos(self):
        return self.book.pos

    def is_current_pos_long(self):
        return self.book.pos == PosType.Long

    def is_current_pos_short(self):
        return self.book.pos == PosType.Short

    def is_current_pos_flat(self):
        return self.book.pos == PosType.Flat

    def set_book_pos(self, pos):
        self.book.pos = pos

    def get_book_price_total(self):
        return self.book.price * self.book.qty

    def get_book_price(self):
        return self.book.price

    def set_book_price(self, price):
        self.book.price = price

    def get_book_qty(self):
        return self.book.qty

    def set_book_qty(self, qty):
        self.book.qty = qty

    def get_book_realized_value(self):
        return self.book.realized_val

    def set_book_realized_value(self, value):
        self.book.previous_realized_val = self.book.realized_val
        self.book.realized_val = value

    def get_book_unrealized_value(self, price):
        qty = self.get_book_qty()
        diff_price = 0
        if self.book.pos == PosType.Long:
            diff_price = price - self.get_book_price()
        elif self.book.pos == PosType.Short:
            diff_price = self.get_book_price() - price

        unr_value = diff_price * qty * self.tick_value / self.tick_size
        return unr_value

    def is_margin_call(self):
        return self.get_book_realized_value() + self.get_book_unrealized_value(
            self.get_current_price()) - self.min_margin <= 0

    def update_book(self, pos, price, qty, realized_val):
        self.set_book_pos(pos)
        self.set_book_price(price)
        self.set_book_qty(qty)
        self.set_book_realized_value(realized_val)

    def get_state(self):
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

        # ADD SENSE OF EPISODE CULMINATION FEATURE. THIS CAN BE TIME STEPS FINISHING UP OR BALANCE
        # FINISHING UP. SINCE BALANCE IS MORE TRANSFERABLE TO REAL TIME, THEN LET'S USE THAT.
        # TRY TO USE ENOUGH $ TO GET THROUGH THE COMPLETE TIME STEP SERIES ON FIRST GO.
        # state = np.append(state, [(self.max_time_steps - self.current_time_step) / self.max_time_steps])
        # REALIZED
        # realized = self.get_book_realized_value()
        # state = np.append(state, [realized / self.init_balance])  # Unrealized

        # UNREALIZED
        # unr = self.get_exit_unrlz_value()
        # state = np.append(state, [unr / self.init_balance])  # Unrealized

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

        # FEATURES
        self.state = np.append(state, self.feature_info[self.current_time_step])

        self.previous_state = self.state

        return self.state

    def tmp_trans_start(self, post_type, in_time_step, last_price, in_price):
        self.tmp_transaction = np.array([post_type, in_time_step, last_price, in_price])

    def get_max_accum_loss(self):
        return self.max_accum_reward - self.max_accum_reward_low_point

    def tmp_trans_close(self, out_time_step, last_price, out_price, pnl):
        post_type = self.tmp_transaction[0]
        in_time_step = self.tmp_transaction[1]
        in_price = self.tmp_transaction[3]
        num_time_steps = out_time_step - in_time_step
        commission = 2 * self.commission_per_order
        net_pnl = pnl - commission
        ticks = (out_price - in_price) / self.tick_size
        drawdown = self.max_drawdown

        # Calculate max accumulated reward
        tot_pnl = self.get_tot_tran_pnl()
        if tot_pnl > self.max_accum_reward:
            self.max_accum_reward = tot_pnl
            self.max_accum_reward_low_point = tot_pnl
        else:
            if tot_pnl < self.max_accum_reward_low_point:
                self.max_accum_reward_low_point = tot_pnl

        self.tmp_transaction = np.array(
            [post_type, in_time_step, last_price, in_price, out_time_step, last_price, out_price, num_time_steps,
             pnl, net_pnl, ticks, commission, drawdown])
        self.transactions = np.vstack([self.transactions, self.tmp_transaction])

        self.tmp_transaction = np.empty(shape=(0, self.transaction_dim))

    def print_transactions(self):
        for i in range(self.transactions.shape[0]):
            t = self.transactions[i]
            pos = ''
            if t[0] == PosType.Short:
                pos = 'S'
            elif t[0] == PosType.Long:
                pos = 'L'
            print("Pos:{}, In_tt:{}, In_Last:{:.2f}, In_Price:{:.2f}, "
                  "Out_tt:{}, Out_Last:{:.2f}, Out_Price:{:.2f}, TT:{}, Pnl:{:.2f}, Net_Pnl:{:.2f}, Ticks:{:.2f}, Comm:{:.2f}".
                  format(pos, t[1], t[2], t[3], t[4], t[5], t[6], t[7], t[8], t[9], t[10], t[11]))

    def get_extra_tick(self):
        return 0

    def add_execution(self, time_step, action, qty, fill_price):
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

        pos_type = action
        pos_qty = qty
        pos_avg_price = fill_price
        if self.executions.shape[0] > 0:
            # Get Last Position info
            last_time_step = self.executions[-1,0]
            last_pos = self.executions[-1, 4]
            last_pos_qty = self.executions[-1, 5]
            last_pos_avg_price = self.executions[-1, 6]

            if (action == PosType.Long and last_pos == PosType.Long) or \
                    (action == PosType.Short and last_pos == PosType.Short):
                pos_type = PosType.Long
                pos_qty = last_pos_qty + qty
                pos_avg_price = (qty*fill_price + last_pos_qty*last_pos_avg_price) / (qty+last_pos_qty)
            elif action == PosType.Long and last_pos == PosType.Short:
                if qty>last_pos_qty:
                    pos_type = PosType.Long
                    pos_qty = qty - last_pos_qty
                    pos_avg_price = fill_price
                elif qty<last_pos_qty:
                    pos_type = PosType.Short
                    pos_qty = last_pos_qty - qty
                    pos_avg_price = last_pos_avg_price
                else: # qty==last_pos_qty:
                    pos_type = PosType.Flat
                    pos_qty = 0
                    pos_avg_price = 0
            elif action == PosType.Short and last_pos == PosType.Long:
                if qty>last_pos_qty:
                    pos_type = PosType.Short
                    pos_qty = qty - last_pos_qty
                    pos_avg_price = fill_price
                elif qty<last_pos_qty:
                    pos_type = PosType.Long
                    pos_qty = last_pos_qty - qty
                    pos_avg_price = last_pos_avg_price
                else: # qty==last_pos_qty:
                    pos_type = PosType.Flat
                    pos_qty = 0
                    pos_avg_price = 0

        # Let's make sure we don't have any overlapping positions.
        assert pos_qty == 1 or pos_qty == 0

        execution = np.array([time_step, action, qty, fill_price, pos_type, pos_qty, pos_avg_price])
        self.executions = np.vstack([self.executions, execution])

    def get_execution_unrlz_step_value(self):
        is_last = False
        i_last = -1
        if self.executions[i_last, 4] == PosType.Flat and self.executions.shape[0] > 1 and self.executions[i_last, 0] == self.current_time_step:
            i_last = -2
            is_last = True
        last_time_step = self.executions[i_last, 0]
        last_pos = self.executions[i_last, 4]
        last_pos_qty = self.executions[i_last, 5]
        last_pos_avg_price = self.executions[i_last, 6]

        unrlzd_step_value = 0
        if last_pos is not PosType.Flat:
            if last_time_step < self.current_time_step:
                if last_pos == PosType.Long:
                    unrlzd_step_value += (self.get_current_bid()-self.get_current_bid(time_step=self.current_time_step-1))*last_pos_qty*self.tick_value/self.tick_size
                elif last_pos == PosType.Short:
                    unrlzd_step_value += (self.get_current_ask(time_step=self.current_time_step-1)-self.get_current_ask())*last_pos_qty*self.tick_value/self.tick_size
            else:
                # charge the commission at the begining of the order
                unrlzd_step_value += - self.commission_per_order * 2
                if last_pos == PosType.Long:
                    unrlzd_step_value += (self.get_current_bid()-last_pos_avg_price)*last_pos_qty*self.tick_value/self.tick_size
                elif last_pos == PosType.Short:
                    unrlzd_step_value += (last_pos_avg_price - self.get_current_ask())*last_pos_qty*self.tick_value/self.tick_size

        self.execution_order_reward += unrlzd_step_value

        if is_last:
            # str1 = "{:.2f}".format(self.execution_order_reward)
            # str2 = "{:.2f}".format(np.sum(self.transactions[-1:, 9]))
            # if str1 != str2:
            #     print("Error detected on step rewards")
            # print("TimeStep {}, Step Reward: {:.2f}, Pnl Rewards {:.2f}".format(self.current_time_step,
            #                                                                     self.execution_order_reward,
            #                                                                     np.sum(self.transactions[-1:, 9])))
            self.execution_order_reward = 0

        return unrlzd_step_value

    def exit_long(self):
        # Update the realized value for 1 share
        new_qty = self.get_book_qty() - 1
        new_price_val = self.get_current_bid()  # - self.get_extra_tick()
        new_unr_val = self.get_book_unrealized_value(new_price_val)
        new_realized_val = self.get_book_realized_value() + new_unr_val - self.commission_per_order
        new_pos = PosType.Flat
        self.position_time_step = 0
        self.inactive_time_step = -1
        self.update_book(new_pos, new_price_val, new_qty, new_realized_val)

        # Close temp transaction and add to transaction history
        self.tmp_trans_close(self.current_time_step, self.get_current_price(), self.get_current_bid(),
                             new_unr_val)

        # Add execution transactions.
        self.add_execution(self.current_time_step, PosType.Short, 1, self.get_current_bid())

    def enter_short(self):
        # Update the realized value for 1 share
        new_qty = self.get_book_qty() + 1
        new_price_val = self.get_current_bid()  # - self.get_extra_tick()
        new_realized_val = self.get_book_realized_value() - self.commission_per_order
        new_pos = PosType.Short
        self.update_book(new_pos, new_price_val, new_qty, new_realized_val)

        # start a transaction
        self.tmp_trans_start(PosType.Short, self.current_time_step, self.get_current_price(), new_price_val)

        # Starting new position
        self.position_time_step = 0
        self.inactive_time_step = 0

        # set drawdown to 0 when entering transaction
        self.max_drawdown = 0

        # Add execution transactions.
        self.add_execution(self.current_time_step, PosType.Short, 1, self.get_current_bid())

    def exit_short(self):
        # Update the realized value for 1 share
        new_qty = self.get_book_qty() - 1
        new_price_val = self.get_current_ask()  # + self.get_extra_tick()
        new_unr_val = self.get_book_unrealized_value(new_price_val)
        new_realized_val = self.get_book_realized_value() + new_unr_val - self.commission_per_order
        new_pos = PosType.Flat
        self.position_time_step = 0
        self.inactive_time_step = -1
        self.update_book(new_pos, new_price_val, new_qty, new_realized_val)
        transaction_closed = True

        # Close temp transaction and add to transaction history
        self.tmp_trans_close(self.current_time_step, self.get_current_price(), self.get_current_ask(),
                             new_unr_val)

        # Add execution transactions.
        self.add_execution(self.current_time_step, PosType.Long, 1, self.get_current_ask())

    def enter_long(self):
        # Update the realized value for 1 share
        new_qty = self.get_book_qty() + 1
        new_price_val = self.get_current_ask()  # + self.get_extra_tick()
        new_realized_val = self.get_book_realized_value() - self.commission_per_order
        new_pos = PosType.Long
        self.update_book(new_pos, new_price_val, new_qty, new_realized_val)

        # start a transaction
        self.tmp_trans_start(PosType.Long, self.current_time_step, self.get_current_price(), new_price_val)

        # Starting new position
        self.position_time_step = 0
        self.inactive_time_step = 0

        # set drawdown to 0 when entering transaction
        self.max_drawdown = 0

        # Add execution transactions.
        self.add_execution(self.current_time_step, PosType.Long, 1, self.get_current_ask())

    def print_transaction_summary(self, session_expired):
        try:
            if session_expired:
                self.session_count += 1
                self.last_session_printed = True
            # avg_num_tt = np.mean(self.transactions[:, 7])
            max_num_tt = np.max(self.transactions[:, 7])
            # avg_pnl = np.mean(self.transactions[:, 9])
            tot_pnl = np.sum(self.transactions[:, 9])
            avg_win = np.mean(self.transactions[:, 9][np.where(self.transactions[:, 9] >= 0)])
            avg_win_tt = np.mean(self.transactions[:, 7][np.where(self.transactions[:, 9] >= 0)])
            avg_loss = np.mean(self.transactions[:, 9][np.where(self.transactions[:, 9] < 0)])
            avg_loss_tt = np.mean(self.transactions[:, 7][np.where(self.transactions[:, 9] < 0)])
            time_step = self.transactions[-1][4]
            tot_trans = self.transactions.shape[0]
            num_wins = np.array(np.where(self.transactions[:, 9] > 0)).shape[1]
            num_losses = np.asarray(np.where(self.transactions[:, 9] < 0)).shape[1]
            num_wins_long = \
                np.array(np.where((self.transactions[:, 9] > 0) & (self.transactions[:, 0] == PosType.Long))).shape[
                    1]
            num_losses_long = \
                np.array(np.where((self.transactions[:, 9] < 0) & (self.transactions[:, 0] == PosType.Long))).shape[
                    1]
            num_wins_short = \
                np.array(
                    np.where((self.transactions[:, 9] > 0) & (self.transactions[:, 0] == PosType.Short))).shape[1]
            num_losses_short = \
                np.array(
                    np.where((self.transactions[:, 9] < 0) & (self.transactions[:, 0] == PosType.Short))).shape[1]
            num_long = self.transactions[np.where(self.transactions[:, 0] == PosType.Long)].shape[0]
            num_short = self.transactions[np.where(self.transactions[:, 0] == PosType.Short)].shape[0]
            # avg_drawdown = np.mean(self.transactions[:, 12])
            # max_drawdown = np.min(self.transactions[:, 12])
            unr_val = 0
            if not self.is_current_pos_flat():
                unr_val = self.get_book_unrealized_value(self.get_current_price())

            print("TimeStep: {}, TotTrans: {}, TotPnl: {:.2f}, Unrlzd:{:.2f}, MaxTS: {}, "
                  "WinTS: {:.2f}, LossTS: {:.2f}, AvgWin: {:.2f}, AvgLoss: {:.2f}, W/L: {}/{}, L/S: {}/{}, "
                  "W/L Long: {}/{}, W/L Short: {}/{}".
                  format(time_step, tot_trans, tot_pnl, unr_val, max_num_tt,
                         avg_win_tt, avg_loss_tt, avg_win, avg_loss, num_wins, num_losses, num_long, num_short,
                         num_wins_long, num_losses_long, num_wins_short, num_losses_short))

            if session_expired:
                print('session {}, final time step {}...'.format(self.session_count, self.current_time_step))

            if self.max_time_steps - self.current_time_step <= 1:
                print('Last step...{}/{}'.format(self.max_time_steps, self.current_time_step))
        except:
            error = True

    def get_tot_tran_pnl(self):
        tot_pnl = 0
        if self.transactions.shape[0] > 0:
            tot_pnl = np.sum(self.transactions[:, 9])
        return tot_pnl

    def print_transaction_summary2(self, session_expired):
        try:
            # print a new line when we start printing for the first time.
            if self.num_print_trans == 0:
                print("\n")
            self.num_print_trans += 1

            if session_expired:
                self.session_count += 1
                self.last_session_printed = True
            # avg_num_tt = np.mean(self.transactions[:, 7])
            max_num_tt = np.max(self.transactions[:, 7])
            # avg_pnl = np.mean(self.transactions[:, 9])
            tot_pnl = np.sum(self.transactions[:, 9])

            num_long = self.transactions[np.where(self.transactions[:, 0] == PosType.Long)].shape[0]
            num_short = self.transactions[np.where(self.transactions[:, 0] == PosType.Short)].shape[0]
            avg_long_win = 0
            avg_long_win_tt = 0
            avg_long_loss = 0
            avg_long_loss_tt = 0
            avg_short_win = 0
            avg_short_win_tt = 0
            avg_short_loss = 0
            avg_short_loss_tt = 0
            if num_long > 0:
                trans = self.transactions[:, 9][
                    np.where((self.transactions[:, 9] >= 0) & (self.transactions[:, 0] == PosType.Long))]
                if len(trans) > 0: avg_long_win = np.mean(trans)

                trans = self.transactions[:, 7][
                    np.where((self.transactions[:, 9] >= 0) & (self.transactions[:, 0] == PosType.Long))]
                if len(trans) > 0: avg_long_win_tt = np.mean(trans)

                trans = self.transactions[:, 9][
                    np.where((self.transactions[:, 9] < 0) & (self.transactions[:, 0] == PosType.Long))]
                if len(trans) > 0: avg_long_loss = np.mean(trans)

                trans = self.transactions[:, 7][
                    np.where((self.transactions[:, 9] < 0) & (self.transactions[:, 0] == PosType.Long))]
                if len(trans) > 0: avg_long_loss_tt = np.mean(trans)
            if num_short > 0:
                trans = self.transactions[:, 9][
                    np.where((self.transactions[:, 9] >= 0) & (self.transactions[:, 0] == PosType.Short))]
                if len(trans) > 0: avg_short_win = np.mean(trans)

                trans = self.transactions[:, 7][
                    np.where((self.transactions[:, 9] >= 0) & (self.transactions[:, 0] == PosType.Short))]
                if len(trans) > 0: avg_short_win_tt = np.mean(trans)

                trans = self.transactions[:, 9][
                    np.where((self.transactions[:, 9] < 0) & (self.transactions[:, 0] == PosType.Short))]
                if len(trans) > 0: avg_short_loss = np.mean(trans)

                trans = self.transactions[:, 7][
                    np.where((self.transactions[:, 9] < 0) & (self.transactions[:, 0] == PosType.Short))]
                if len(trans) > 0: avg_short_loss_tt = np.mean(trans)

            time_step = self.transactions[-1][4]
            tot_trans = self.transactions.shape[0]
            num_wins_long = \
                np.array(np.where((self.transactions[:, 9] > 0) & (self.transactions[:, 0] == PosType.Long))).shape[
                    1]
            num_losses_long = \
                np.array(np.where((self.transactions[:, 9] < 0) & (self.transactions[:, 0] == PosType.Long))).shape[
                    1]
            num_wins_short = \
                np.array(
                    np.where((self.transactions[:, 9] > 0) & (self.transactions[:, 0] == PosType.Short))).shape[1]
            num_losses_short = \
                np.array(
                    np.where((self.transactions[:, 9] < 0) & (self.transactions[:, 0] == PosType.Short))).shape[1]
            # avg_drawdown = np.mean(self.transactions[:, 12])
            # max_drawdown = np.min(self.transactions[:, 12])
            unr_val = 0
            if not self.is_current_pos_flat():
                unr_val = self.get_book_unrealized_value(self.get_current_price())

            pos = ""
            if self.is_current_pos_long(): pos = "L"
            if self.is_current_pos_short(): pos = "S"

            print("TimeStep:{:>7}, TotTrans:{:>4}, TotPnl:{:6.2f}, Unrlzd:{}{:5.2f}, MaxTS:{:>5}, "
                  "L-Win/Loss:{:5.2f}/{:5.2f}, L-TS-W/L:{:5.2f}/{:5.2f}, "
                  "S-Win/Loss:{:5.2f}/{:5.2f}, S-TS-W/L:{:5.2f}/{:5.2f}, "
                  "L/S:{:>5}/{:>5}, "
                  "W/L Long:{:>5}/{:>5}, W/L Short:{:>5}/{:>5}".
                  format(time_step, tot_trans, tot_pnl, pos, unr_val, max_num_tt,
                         avg_long_win, avg_long_loss, avg_long_win_tt, avg_long_loss_tt,
                         avg_short_win, avg_short_loss, avg_short_win_tt, avg_short_loss_tt,
                         num_long, num_short,
                         num_wins_long, num_losses_long, num_wins_short, num_losses_short))

            if session_expired:
                print('session {}, final time step {}...'.format(self.session_count, self.current_time_step))

            if self.max_time_steps - self.current_time_step <= 1:
                print('Last step...{}/{}'.format(self.max_time_steps, self.current_time_step))
        except:
            error = True

    def get_last_transaction(self):
        last_pos = 0
        last_return = 0
        if self.transactions.shape[0] > 0:
            pos = self.transactions[-1, 0]
            if pos == PosType.Long: last_pos = 1
            last_return = self.transactions[-1, 9] / self.max_return_norm

        return last_pos, last_return

    def is_same_entry_as_previous(self, pos):
        pos_val = 0
        if pos == PosType.Long:
            pos_val = 1

        last_pos, last_return = self.get_last_transaction()
        ret = False
        if pos_val == last_pos and last_return < 0:
            ret = True
        return ret

    def get_discounted_unrlzd_value(self):
        fv = 0
        r = 0.03
        i_tran = len(self.tran_unrlz_value)
        if i_tran == 1:
            fv = self.tran_unrlz_value[i_tran - 1]
        elif i_tran > 1:
            fv = self.tran_unrlz_value[i_tran - 1] - self.tran_unrlz_value[i_tran - 2]

        pv = fv / (1 + r) ** i_tran

        return pv

    def get_exit_unrlz_value(self, time_step=-1):
        if time_step == -1:
            time_step = self.current_time_step
        exit_unrlz = 0
        commission = self.commission_per_order
        if self.is_current_pos_long():
            exit_unrlz = self.get_book_unrealized_value(self.get_current_bid(time_step=time_step))
        elif self.is_current_pos_short():
            exit_unrlz = self.get_book_unrealized_value(self.get_current_ask(time_step=time_step))
        else:
            commission = 0
        net_unrlz = exit_unrlz - commission  # Just the exit commission.
        return net_unrlz

    def step_flat(self):
        if self.is_current_pos_flat():
            self.current_time_step += 1

    @deprecated(version='1.0.0', reason="This funciton was used when FeatureList contained  Elliott single pivots")
    def enter_action_votes(self, current_time_step=None, n_features=30):
        if current_time_step is None:
            current_time_step = self.current_time_step
        features = self.feature_info[:, 0:n_features]
        cnt = features.shape[1]
        sum = (cnt / 2) * (1 + cnt)
        votes_long = 0
        votes_short = 0
        for i in range(cnt):
            vote = (i + 1) / sum
            zigzag = features[current_time_step, i]
            if zigzag >= 0.5:
                votes_long += vote
            else:
                votes_short += vote
        return votes_long, votes_short

    @deprecated(version='1.0.0', reason="This funciton was used when FeatureList contained  Elliott single pivots")
    def enter_action(self):
        votes_long, votes_short = self.enter_action_votes(self.current_time_step)
        if votes_short <= votes_long:
            action = 1
        else:
            action = 0
        return action

    def plot_episode_returns(self):
        plt.plot(self.episode_rewards)
        plt.xlabel = 'Episode'
        plt.ylabel = 'PnL'
        plt.show()

    def plot_price_and_returns(self):
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
            if self.transactions[i][0] == PosType.Long:
                axs[0].plot([x1, x2], [y1, y2], 'g')
            else:
                axs[0].plot([x1, x2], [y1, y2], 'r')

        # ax1.plot(x, y, color=color)
        axs[0].tick_params(axis='y', labelcolor=color)

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
        axs[1].set_xlabel("Time Step. Episode {}".format(len(self.episode_rewards)))
        axs[1].xaxis.label.set_color('cyan')
        axs[1].tick_params(axis='x', colors='cyan')
        axs[1].set_ylabel('Price', color=color)  # we already handled the x-label with ax1
        axs[1].plot(x2, y2, color='lightgray')

        for i in range(self.transactions.shape[0]):
            x1 = self.transactions[i][1]
            x2 = self.transactions[i][4]
            y1 = self.transactions[i][3]
            y2 = self.transactions[i][5]
            if self.transactions[i][0] == PosType.Long:
                axs[1].plot([x1, x2], [y1, y2], 'g')
            else:
                axs[1].plot([x1, x2], [y1, y2], 'r')

        axs[1].tick_params(axis='y', labelcolor=color)

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        fig.set_size_inches(11, 8)
        plt.show()

    def step_fast_forward(self):
        # prevent from getting called back while looping.
        self.ff = True

        stop_ff = False
        state, ret_reward, info = None, None, None
        while not stop_ff:
            state, ret_reward, self.done, info = self.step(-1)
            unrlz = self.get_exit_unrlz_value()
            if self.done or unrlz > 0 or unrlz <= -self.min_ff_loss or self.is_current_pos_flat():
                stop_ff = True

        self.ff = False
        return state, ret_reward, self.done, info

    def skip(self, no_steps=1000):
        self.ff = True

        state, ret_reward, info = None, None, None
        for i in range(no_steps):
            state, ret_reward, self.done, info = self.step(-1)
            if self.done or self.current_time_step == self.feature_info.shape[0] - 1:
                break

        self.ff = False
        return state, ret_reward, self.done, info

    def step(self,
             action,
             ):

        self.min_ff_loss = self.min_ff_loss
        # If we want to train a side, we will just modify the action so that it skips when not going
        # in one direction
        if not self.ff and self.train_side is not None and self.is_current_pos_flat():
            if action != self.train_side and self.skip_steps >= 1:
                # action = -1
                state, ret_reward, self.done, info = self.skip(self.skip_steps)
                if self.normalize_return:
                    ret_reward = (ret_reward + self.skip_incentive) / self.max_return_norm
                else:
                    ret_reward = ret_reward + self.skip_incentive
                return state, ret_reward, self.done, info
            elif action != self.train_side and self.skip_steps == 0:
                action = self.train_side

        self.incentives = 0

        minutes_left_in_session = self.featureList[self.current_time_step][3]
        session_expired = (minutes_left_in_session <= 1) or (self.max_time_steps - self.current_time_step <= 1)
        if not session_expired:
            self.last_session_printed = False

        # CHECK FOR MAX PROFIT OR MAX DRAW DOWN WAS HIT TO EXIT
        exit_target = self.take_profit
        exit_stop = self.take_loss
        exit_target_flag = (self.get_book_unrealized_value(self.get_current_price()) >= exit_target)
        exit_stop_flag = (self.get_book_unrealized_value(self.get_current_price()) <= -exit_stop)
        exit_pos = exit_target_flag or exit_stop_flag

        # CHECK FOR END OF TIME STEPS TO EXIT POSITION
        if session_expired and (not self.is_current_pos_flat()):
            exit_pos = True
            print('Exiting none flat position on last time step...{}'.format(self.current_time_step))

        # if (exit_target_flag):
        #     print("Took Profit...")
        # if (exit_stop_flag):
        #     print("Forced Exit Stop Loss...")

        # CHECK FOR MAX PROFIT OR MAX DRAW DOWN WAS HIT TO EXIT
        # exit_pos = (action == 2 and self.get_book_unrealized_value(self.get_current_price()) >= 200) or \
        #            (action == 2 and self.get_book_unrealized_value(self.get_current_price()) <= -500)
        # if exit_pos:
        #     print("Forced Exit")

        tran_closed = False
        # net_unrlz = self.get_exit_unrlz_value()
        tt = self.position_time_step
        can_exit = True  # net_unrlz > 0 or net_unrlz < -900 # or tt >= 30
        if action == 1 and self.is_current_pos_flat() and (not session_expired):  # ENTER LONG
            self.enter_long()
            self.tran_unrlz_value = []
            if self.is_same_entry_as_previous(PosType.Long):
                self.incentives += self.same_entry_penalty

        elif (action == 1 and self.is_current_pos_short() and can_exit) or (self.is_current_pos_short() and exit_pos):  # EXIT SHORT
            self.exit_short()
            tran_closed = True
        elif action == 0 and self.is_current_pos_flat() and (not session_expired):  # ENTER SHORT
            self.enter_short()
            self.tran_unrlz_value = []
            if self.is_same_entry_as_previous(PosType.Short):
                self.incentives += self.same_entry_penalty
        elif (action == 0 and self.is_current_pos_long() and can_exit) or (self.is_current_pos_long() and exit_pos):  # EXIT LONG
            self.exit_long()
            tran_closed = True

        self.done = self.is_margin_call() or self.get_book_realized_value() <= 0 or \
                    (self.max_time_steps - self.current_time_step <= 1)

        # If transaction is still open and episode is done, we need to close the transaction
        if self.done and not self.is_current_pos_flat():
            if self.is_current_pos_long():
                self.exit_long()
                tran_closed = True
            elif self.is_current_pos_short():
                self.exit_short()
                tran_closed = True

        # REWARD
        reward = self.get_book_realized_value() - self.book.previous_realized_val

        # Calculate exit reward for each step
        if self.reward_for_each_step:
            reward = self.get_execution_unrlz_step_value()

        # INCENTIVES AND PENALTIES (NOT BEING USED)
        # if self.inactive_time_step > 100:
        #     self.incentives += 0
        #
        # if self.position_time_step > 1000:
        #     self.incentives += 0  # -25
        #
        # if 0 < reward < 70:
        #     self.incentives += 0  # 100
        #
        # if self.get_book_unrealized_value(self.get_current_price()) < 0:
        #     self.incentives += 0  # -.5
        # elif self.get_book_unrealized_value(self.get_current_price()) > 0:
        #     self.incentives += 0  # .5

        # Adjust MAX Drawdown
        if not self.is_current_pos_flat():
            self.max_drawdown = min(self.max_drawdown, self.get_book_unrealized_value(self.get_current_price()))

        state = self.get_state()

        # NOT BEING USED
        # margin = self.get_book_realized_value() + self.get_book_unrealized_value(
        #     self.get_current_price()) - self.min_margin

        # or self.position_time_step % 500 == 0
        if (session_expired and not self.last_session_printed) or (
                (self.current_time_step % 500 == 0) and self.transactions.shape[0] > 0):
            self.print_transaction_summary2(session_expired)

        info = 0
        self.set_book_realized_value(self.get_book_realized_value())

        if not self.is_current_pos_flat() and not self.done:
            self.tran_unrlz_value.append(self.get_book_unrealized_value(self.get_current_price()))

        ret_reward = reward + self.incentives
        if self.normalize_return:
            ret_reward = ret_reward / self.max_return_norm

        if self.done:
            self.episode_rewards = np.append(self.episode_rewards, self.get_tot_tran_pnl())
            self.save_episode_rewards()
            # Safer if you plot from Agent after saving the model.
            # self.plot_episode_returns()
            # self.plot_price_and_returns()

        # Move one time step
        self.current_time_step += 1

        # Move inactive or position time step
        if self.is_current_pos_flat():
            self.inactive_time_step += 1
        else:
            self.position_time_step += 1

        # This will fast forward the steps until we are in a positive exit position or until min loss
        # threshold has been hit.
        exit_unrlz = self.get_exit_unrlz_value()
        if self.ff_on_loss and not self.done and not self.ff and not self.is_current_pos_flat() and exit_unrlz < 0:
            return self.step_fast_forward()

        return state, ret_reward, self.done, info

    def render(self, mode='human', close=False):
        # Renders the environment to the screen
        # This should replace the printing.
        print_nothing = True
