import pandas as pd
import numpy as np

class ArbitrageAlphaModel(AlphaModel):
    """
    This class monitors the intraday bid and ask prices of two correlated ETFs. When the bid price of ETF A (B) diverts 
    high enough away from the ask price of ETF B (A) such that the profit_pct_threshold is reached, we start a timer. If 
    the arbitrage opportunity is still present after the specified timesteps, we enter the arbitrage trade by going long 
    ETF B (A) and short ETF A (B). When the spread reverts back to where the bid of ETF B (A) >= the ask of ETF A (B) for 
    the same number of timesteps, we exit the trade. To address a trending historical spread between the two ETFs, we 
    adjust the spread by removing the mean spread over a lookback window.
    """
    symbols = [] # IVV, SPY
    entry_timer = [0, 0]
    exit_timer = [0, 0]
    spread_adjusters = [0, 0]
    long_side = -1
    consolidators = {}
    history = {}
    
    def __init__(self, order_delay = 3, profit_pct_threshold = 0.02, window_size = 400):
        """
        Input:
         - order_delay
            The number of timesteps to wait while an arbitrage opportunity is present before emitting insights
            (>= 0)
         - profit_pct_threshold
            The amount of adjusted profit there must be in an arbitrage opportunity to signal a potential entry
            (> 0)
         - window_size
            The length of the lookback window used to calculate the spread adjusters (to address the trending spread b/w ETFs)
            (> 0)
        """
        self.order_delay = order_delay
        self.pct_threshold = profit_pct_threshold / 100
        self.window_size = window_size
        self.consolidated_update = 0
    
    
    def Update(self, algorithm, data):
        """
        Called each time our alpha model receives a new data slice.
        
        Input:
         - algorithm
            Algorithm instance running the backtest
         - data
            Data for the current time step in the backtest
            
        Returns a list of Insights to the portfolio construction model.
        """
        if algorithm.IsWarmingUp:
            return []
        
        quotebars = self.get_quotebars(data)
        if not quotebars:
            return []

        # Ensure we are not within 5 minutes of either the open or close
        exchange = algorithm.Securities['SPY'].Exchange
        if not (exchange.DateTimeIsOpen(algorithm.Time - timedelta(minutes=5)) and \
                exchange.DateTimeIsOpen(algorithm.Time + timedelta(minutes=5))):
            return []
        
        # Search for entries
        for i in range(2):
            if quotebars[abs(i-1)].Bid.Close / quotebars[i].Ask.Close - self.spread_adjusters[abs(i-1)] >= self.pct_threshold:
                self.entry_timer[i] += 1
                if self.entry_timer[i] == self.order_delay:
                    self.exit_timer = [0, 0]
                    if self.long_side == i:
                        return []
                    self.long_side = i
                    return [Insight.Price(self.symbols[i], timedelta(days=9999), InsightDirection.Up),
                            Insight.Price(self.symbols[abs(i-1)], timedelta(days=9999), InsightDirection.Down)]
                else:
                    return []
            self.entry_timer[i] = 0
            
        # Search for an exit
        if self.long_side >= 0: # In a position
            if quotebars[self.long_side].Bid.Close / quotebars[abs(self.long_side-1)].Ask.Close - self.spread_adjusters[self.long_side] >= 0: # Exit signal
                self.exit_timer[self.long_side] += 1
                if self.exit_timer[self.long_side] == self.order_delay: # Exit signal lasted long enough
                    self.exit_timer[self.long_side] = 0
                    i = self.long_side
                    self.long_side = -1
                    return [Insight.Price(self.symbols[i], timedelta(days=9999), InsightDirection.Flat),
                            Insight.Price(self.symbols[abs(i-1)], timedelta(days=9999), InsightDirection.Flat)]
                else:
                    return []
        return []
        
        
    def OnSecuritiesChanged(self, algorithm, changes):
        """
        Called each time our universe has changed.
        
        Inputs:
         - algorithm
            Algorithm instance running the backtest
         - changes
            The additions and subtractions to the algorithm's security subscriptions
        """
        added_symbols = [security.Symbol for security in changes.AddedSecurities]
        if len(added_symbols) > 0:
            self.symbols.extend(added_symbols)
            
            if len(self.symbols) != 2:
                algorithm.Error(f"ArbitrageAlphaModel must have 2 symbols to trade")
                algorithm.Quit()
                return
            
            history = algorithm.History(self.symbols, self.window_size, Resolution.Second)[['bidclose', 'askclose']]
            starting_row_count = min([history.loc[symbol].shape[0] for symbol in self.symbols])

            for symbol in self.symbols:
                self.history[symbol] = {'bids': history.loc[symbol].bidclose.to_numpy()[-starting_row_count:],
                                        'asks': history.loc[symbol].askclose.to_numpy()[-starting_row_count:]}

            self.update_spread_adjusters()
            
            # Setup daily consolidators to update spread_adjusters
            for symbol in self.symbols:
                self.consolidators[symbol] = QuoteBarConsolidator(1)
                self.consolidators[symbol].DataConsolidated += self.CustomDailyHandler
                algorithm.SubscriptionManager.AddConsolidator(symbol, self.consolidators[symbol])
            
        for removed in changes.RemovedSecurities:
            algorithm.SubscriptionManager.RemoveConsolidator(removed.Symbol, self.consolidators[removed.Symbol])
    
    
    def CustomDailyHandler(self, sender, consolidated):
        """
        Updates the rolling lookback window with the latest data.
        
        Inputs
         - sender
            Function calling the consolidator
         - consolidated
            Tradebar representing the latest completed trading day
        """
        # Add new data point to history while removing expired history
        self.history[consolidated.Symbol]['bids'] = np.append(self.history[consolidated.Symbol]['bids'][-self.window_size:], consolidated.Bid.Close)
        self.history[consolidated.Symbol]['asks'] = np.append(self.history[consolidated.Symbol]['asks'][-self.window_size:], consolidated.Ask.Close)
        
        # After updating the history of both symbols, update the spread adjusters
        self.consolidated_update += 1
        if self.consolidated_update == 2:
            self.consolidated_update = 0
            self.update_spread_adjusters()
        


    def get_quotebars(self, data):
        """
        Extracts the QuoteBars from the given slice.
        
        Inputs
         - data
            Latest slice object the algorithm has received
            
        Returns the QuoteBars for the symbols we are trading.
        """
        if not all([data.QuoteBars.ContainsKey(symbol) for symbol in self.symbols]):
            return []
            
        quotebars = [data.QuoteBars[self.symbols[i]] for i in range(2)]
        
        if not all([q is not None for q in quotebars]):
            return []
            
        # Ensure ask > bid for each ETF
        if not all([q.Ask.Close > q.Bid.Close for q in quotebars]):
            return []
            
        return quotebars
    
    
    def update_spread_adjusters(self):
        """
        Updates the spread adjuster by finding the mean of the trailing spread ratio b/w ETFs.
        """
        for i in range(2):
            numerator_history = self.history[self.symbols[i]]['bids']
            denominator_history = self.history[self.symbols[abs(i-1)]]['asks']
            self.spread_adjusters[i] = (numerator_history / denominator_history).mean()
