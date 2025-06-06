import logging
from config import MAX_POSITION_RATIO
import time
import numpy as np
import asyncio  # 添加asyncio导入用于锁机制
import aiohttp  # 添加aiohttp用于异步HTTP请求

class AdvancedRiskManager:
    def __init__(self, trader):
        self.trader = trader
        self.logger = logging.getLogger(self.__class__.__name__)
        # 初始化恢复相关属性
        self.recovery_cooldown_until = 0
        self.recovery_executed = False
        self.last_recovery_time = 0
        self.last_position_ratio = 0
        # 添加待执行买入任务状态
        self.pending_recovery = False
        self.pending_recovery_amount = 0
        self.pending_recovery_target_price = 0
        self.pending_recovery_price_expiry = 0
        
        # 添加状态锁，防止并发操作导致的状态不一致
        self.state_lock = asyncio.Lock()
        
        # 记录最后已知价格，用于API故障恢复
        self.last_known_price = None
        
        # 添加技术指标缓存和更新时间记录
        self.indicator_cache = {
            # RSI缓存
            'rsi_4h': {'value': None, 'last_update': 0},
            'rsi_1d': {'value': None, 'last_update': 0},
            # 布林带缓存
            'bb_4h': {'value': (None, None, None), 'last_update': 0},
            'bb_1d': {'value': (None, None, None), 'last_update': 0},
            # 支撑位缓存
            'support_levels': {'value': [], 'last_update': 0},
            # MA数据缓存
            'ma_data': {'value': (None, None), 'last_update': 0},
            # MACD数据缓存
            'macd_data': {'value': (None, None), 'last_update': 0},
            # 价格分位数缓存
            'price_percentile': {'value': None, 'last_update': 0},
            # 短期趋势缓存
            'short_trend': {'value': (0, None), 'last_update': 0},
            # K线数据缓存
            'klines_1h': {'value': None, 'last_update': 0},
            'klines_4h': {'value': None, 'last_update': 0},
            'klines_1d': {'value': None, 'last_update': 0},
            # 恐慌贪婪指数缓存
            'fear_greed_index': {'value': 50, 'last_update': 0},
        }
        
        # 添加评分历史和仓位比例历史
        self.recent_scores = []  # 存储最近的评分历史
        self.position_ratio_history = []  # 存储最近的仓位比例历史
        
        # 设置不同指标的更新间隔（秒）
        self.update_intervals = {
            'rsi_4h': 15 * 60,  # 15分钟
            'rsi_1d': 60 * 60,  # 1小时
            'bb_4h': 15 * 60,   # 15分钟
            'bb_1d': 60 * 60,   # 1小时
            'support_levels': 4 * 60 * 60,  # 4小时
            'ma_data': 60 * 60,  # 1小时
            'macd_data': 60 * 60,  # 1小时
            'price_percentile': 4 * 60 * 60,  # 4小时
            'short_trend': 15 * 60,  # 15分钟
            'klines_1h': 5 * 60,  # 5分钟
            'klines_4h': 15 * 60,  # 15分钟
            'klines_1d': 60 * 60,  # 1小时
            'fear_greed_index': 8 * 60 * 60,  # 8小时，因为指数通常每天更新一次
        }
    
    async def multi_layer_check(self):
        # self.logger.info("▶ 风控检查")
        try:
            # 使用状态锁保护共享状态
            async with self.state_lock:
                position_ratio = await self._get_position_ratio()
                
                # 保存上次的仓位比例
                if not hasattr(self, 'last_position_ratio'):
                    self.last_position_ratio = position_ratio
                
                # 只在仓位比例变化超过0.1%时打印日志
                if abs(position_ratio - self.last_position_ratio) > 0.001:
                    self.logger.info(
                        f"风控检查 | "
                        f"当前仓位比例: {position_ratio:.2%} | "
                        f"最大允许比例: {self.trader.config.MAX_POSITION_RATIO:.2%} | "
                        f"最小底仓比例: {self.trader.config.MIN_POSITION_RATIO:.2%}"
                    )
                    self.last_position_ratio = position_ratio
                    
                if position_ratio < self.trader.config.MIN_POSITION_RATIO:
                    self.logger.warning(f"底仓保护触发 | 当前: {position_ratio:.2%}")
                    
                    asyncio.create_task(self._safe_check_pending_recovery_main())
                    
                    return True
                
                if position_ratio > self.trader.config.MAX_POSITION_RATIO:
                    self.logger.warning(f"仓位超限 | 当前: {position_ratio:.2%}")
                    return True
        except Exception as e:
            self.logger.error(f"风控检查失败: {str(e)}")
            return False

    async def _calculate_rsi(self, period=14, timeframe='4h', klines=None):
        """
        计算当前RSI值
        
        Args:
            period: RSI计算周期，默认14
            timeframe: K线时间周期，默认4小时
            klines: 可选的K线数据，如果提供则使用，否则获取
            
        Returns:
            float: RSI值(0-100)，如果计算失败则返回50
        """
        indicator_key = f'rsi_{timeframe}'
        
        # 检查缓存是否需要更新
        if not klines and not await self._should_update_indicator(indicator_key):
            return self.indicator_cache[indicator_key]['value']
            
        try:
            # 如果没有提供K线数据，则获取
            if not klines:
                klines = await self._get_klines_data(timeframe, period*2, force_update=False)
            
            if not klines or len(klines) < period+1:
                self.logger.warning(f"计算RSI失败: 获取的K线数据不足 ({len(klines) if klines else 0} < {period+1})")
                return 50  # 返回中性RSI值
                
            # 提取收盘价
            closes = np.array([float(kline[4]) for kline in klines])
            
            # 计算价格变化
            deltas = np.diff(closes)
            
            # 分离上涨和下跌
            gains = deltas.copy()
            losses = deltas.copy()
            gains[gains < 0] = 0
            losses[losses > 0] = 0
            losses = abs(losses)
            
            # 计算初始平均值
            avg_gain = np.mean(gains[:period])
            avg_loss = np.mean(losses[:period])
            
            # 使用Wilder平滑RSI计算方法
            for i in range(period, len(deltas)):
                avg_gain = (avg_gain * (period - 1) + gains[i]) / period
                avg_loss = (avg_loss * (period - 1) + losses[i]) / period
            
            # 计算相对强度RS
            if avg_loss == 0:
                rsi = 100  # 避免除以零
            else:
                rs = avg_gain / avg_loss
                # 计算RSI
                rsi = 100 - (100 / (1 + rs))
            
            self.logger.debug(f"RSI({period}, {timeframe}): {rsi:.2f}")
            
            # 更新缓存
            self.indicator_cache[indicator_key] = {
                'value': rsi,
                'last_update': time.time()
            }
            
            return rsi
            
        except Exception as e:
            self.logger.error(f"计算RSI失败: {str(e)}")
            return 50  # 返回中性RSI值

    async def _calculate_bollinger_bands(self, period=20, deviation=2, timeframe='4h', klines=None):
        """
        计算布林带上中下轨
        
        Args:
            period: 移动平均周期，默认20
            deviation: 标准差倍数，默认2
            timeframe: K线时间周期，默认4小时
            klines: 可选的K线数据，如果提供则使用，否则获取
            
        Returns:
            tuple: (upper, middle, lower) 布林带上中下轨
                  如果计算失败则返回(None, None, None)
        """
        indicator_key = f'bb_{timeframe}'
        
        # 检查缓存是否需要更新
        if not klines and not await self._should_update_indicator(indicator_key):
            return self.indicator_cache[indicator_key]['value']
            
        try:
            # 如果没有提供K线数据，则获取
            if not klines:
                klines = await self._get_klines_data(timeframe, period+10, force_update=False)
            
            if not klines or len(klines) < period:
                self.logger.warning(f"计算布林带失败: 获取的K线数据不足 ({len(klines) if klines else 0} < {period})")
                return None, None, None
                
            # 提取收盘价
            closes = np.array([float(kline[4]) for kline in klines])
            
            # 计算移动平均线(中轨)
            middle = np.mean(closes[-period:])
            
            # 计算标准差
            std_dev = np.std(closes[-period:])
            
            # 计算上下轨
            upper = middle + (std_dev * deviation)
            lower = middle - (std_dev * deviation)
            
            current_price = await self.trader._get_latest_price()
            self.logger.debug(
                f"布林带(周期={period}, 偏差={deviation}, {timeframe}): "
                f"上轨={upper:.2f}, "
                f"中轨={middle:.2f}, "
                f"下轨={lower:.2f}, "
                f"当前价={current_price:.2f}, "
                f"位置={(current_price-lower)/(upper-lower) if upper != lower else 0.5:.2f}"
            )
            
            # 更新缓存
            self.indicator_cache[indicator_key] = {
                'value': (upper, middle, lower),
                'last_update': time.time()
            }
            
            return upper, middle, lower
            
        except Exception as e:
            self.logger.error(f"计算布林带失败: {str(e)}")
            return None, None, None

    async def _get_support_levels(self, timeframe='1d', lookback=90, klines=None):
        """
        获取市场关键支撑位
        
        Args:
            timeframe: K线时间周期，默认1天
            lookback: 回溯周期数，默认90
            klines: 可选的K线数据，如果提供则使用，否则获取
            
        Returns:
            list: 支撑位价格列表，按价格从低到高排序
        """
        indicator_key = 'support_levels'
        
        # 检查缓存是否需要更新
        if not klines and not await self._should_update_indicator(indicator_key):
            return self.indicator_cache[indicator_key]['value']
            
        try:
            # 如果没有提供K线数据，则获取
            if not klines:
                klines = await self._get_klines_data(timeframe, lookback, force_update=False)
            
            if not klines or len(klines) < 20:  # 至少需要20根K线
                self.logger.warning(f"获取支撑位失败: 获取的K线数据不足 ({len(klines) if klines else 0} < 20)")
                return []
                
            # 提取低点价格
            lows = np.array([float(kline[3]) for kline in klines])
            
            # 计算布林带下轨作为一个支撑位
            _, _, bb_lower = await self._calculate_bollinger_bands(timeframe=timeframe)
            
            # 计算移动平均线作为支撑位
            ma_50 = np.mean(np.array([float(kline[4]) for kline in klines])[-50:]) if len(klines) >= 50 else None
            ma_200 = np.mean(np.array([float(kline[4]) for kline in klines])[-200:]) if len(klines) >= 200 else None
            
            # 寻找局部低点
            support_levels = []
            window_size = 5  # 前后5根K线范围内的低点判定
            
            for i in range(window_size, len(lows) - window_size):
                if all(lows[i] <= lows[j] for j in range(i-window_size, i)) and \
                   all(lows[i] <= lows[j] for j in range(i+1, i+window_size+1)):
                    support_levels.append(lows[i])
            
            # 添加布林带下轨和移动平均线支撑位
            if bb_lower:
                support_levels.append(bb_lower)
            if ma_50:
                support_levels.append(ma_50)
            if ma_200:
                support_levels.append(ma_200)
                
            # 添加斐波那契回调位作为支撑位
            # 先找出最近的高点和低点
            if len(klines) >= 30:
                recent_high = max([float(kline[2]) for kline in klines[-30:]])
                recent_low = min([float(kline[3]) for kline in klines[-30:]])
                
                # 计算斐波那契回调位
                fib_levels = [0.236, 0.382, 0.5, 0.618, 0.786]
                for level in fib_levels:
                    fib_support = recent_high - (recent_high - recent_low) * level
                    support_levels.append(fib_support)
            
            # 合并相近的支撑位（差异小于1%）
            support_levels.sort()
            merged_supports = []
            
            if support_levels:
                current_support = support_levels[0]
                merged_supports.append(current_support)
                
                for level in support_levels[1:]:
                    # 如果当前支撑位与上一个合并的支撑位相差超过1%，则添加为新的支撑位
                    if (level - current_support) / current_support > 0.01:
                        current_support = level
                        merged_supports.append(current_support)
            
            current_price = await self.trader._get_latest_price()
            
            # 只保留低于当前价格的支撑位
            valid_supports = [s for s in merged_supports if s < current_price * 0.99]  # 略低于当前价格的支撑位
            
            if valid_supports:
                self.logger.debug(f"已识别支撑位: {', '.join([f'{s:.2f}' for s in valid_supports])}")
            else:
                self.logger.debug(f"未找到有效支撑位")
                
            # 更新缓存
            self.indicator_cache[indicator_key] = {
                'value': valid_supports,
                'last_update': time.time()
            }
                
            return valid_supports
            
        except Exception as e:
            self.logger.error(f"获取支撑位失败: {str(e)}")
            return []

    async def _evaluate_entry_conditions(self):
        """
        综合评估入场条件，给出得分和建议
        
        Returns:
            tuple: (score, details, target_price)
                score: 入场条件得分 (0-100)，越高越适合买入
                details: 入场条件详情字典
                target_price: 建议的目标买入价格
        """
        try:
            # 初始化得分和详情
            score = 50  # 中性得分
            details = {}
            
            # 获取当前价格
            current_price = await self._get_latest_price_with_retry()
            if not current_price:
                return 0, {'error': '无法获取当前价格'}, None
                
            # 批量预获取K线数据，减少API请求
            await self._get_klines_data('4h', 50, force_update=False)  # 用于RSI和布林带
            await self._get_klines_data('1d', 90, force_update=False)  # 用于日线指标和支撑位
            await self._get_klines_data('1h', 24, force_update=False)  # 用于短期趋势
                
            # 1. 计算RSI，检查是否超卖 (14周期，4小时和1天两个周期)
            rsi_4h = await self._calculate_rsi(period=14, timeframe='4h')
            rsi_1d = await self._calculate_rsi(period=14, timeframe='1d')
            
            # RSI得分计算 (RSI越低，得分越高)
            rsi_score = 0
            
            if rsi_4h < 30:  # 超卖
                rsi_score += 15
                details['rsi_4h'] = f"{rsi_4h:.1f} (超卖)"
            elif rsi_4h < 40:
                rsi_score += 10
                details['rsi_4h'] = f"{rsi_4h:.1f} (低位)"
            else:
                details['rsi_4h'] = f"{rsi_4h:.1f}"
                
            if rsi_1d < 30:  # 日线超卖，更强信号
                rsi_score += 20
                details['rsi_1d'] = f"{rsi_1d:.1f} (超卖)"
            elif rsi_1d < 40:
                rsi_score += 15
                details['rsi_1d'] = f"{rsi_1d:.1f} (低位)"
            else:
                details['rsi_1d'] = f"{rsi_1d:.1f}"
                
            # 2. 计算布林带，检查是否触及下轨
            bb_4h = await self._calculate_bollinger_bands(period=20, timeframe='4h')
            bb_1d = await self._calculate_bollinger_bands(period=20, timeframe='1d')
            
            # 布林带得分计算
            bb_score = 0
            
            if bb_4h[2] and current_price <= bb_4h[2]:  # 价格触及或跌破4小时布林带下轨
                bb_score += 15
                details['bb_4h'] = f"价格触及下轨 ({bb_4h[2]:.1f})"
            elif bb_4h[2] and current_price <= bb_4h[2] * 1.01:  # 价格接近4小时布林带下轨
                bb_score += 10
                details['bb_4h'] = f"价格接近下轨 ({bb_4h[2]:.1f})"
            else:
                details['bb_4h'] = f"价格在带内 ({bb_4h[1]:.1f})"
                
            if bb_1d[2] and current_price <= bb_1d[2]:  # 价格触及或跌破日线布林带下轨
                bb_score += 20
                details['bb_1d'] = f"价格触及下轨 ({bb_1d[2]:.1f})"
            elif bb_1d[2] and current_price <= bb_1d[2] * 1.01:  # 价格接近日线布林带下轨
                bb_score += 15
                details['bb_1d'] = f"价格接近下轨 ({bb_1d[2]:.1f})"
            else:
                details['bb_1d'] = f"价格在带内 ({bb_1d[1]:.1f})"
                
            # 3. 获取关键支撑位
            support_levels = await self._get_support_levels()
            
            # 支撑位得分计算
            support_score = 0
            
            if support_levels:
                # 寻找最近的支撑位
                nearest_support = max([s for s in support_levels if s < current_price]) if support_levels else None
                
                if nearest_support:
                    # 计算当前价格与最近支撑位的距离
                    distance_percent = (current_price - nearest_support) / nearest_support
                    
                    if distance_percent < 0.01:  # 价格非常接近支撑位 (1%以内)
                        support_score += 20
                        details['support'] = f"价格接近强支撑位 ({nearest_support:.1f})"
                    elif distance_percent < 0.03:  # 价格接近支撑位 (3%以内)
                        support_score += 15
                        details['support'] = f"价格接近支撑位 ({nearest_support:.1f})"
                    else:
                        details['support'] = f"最近支撑位 ({nearest_support:.1f})"
                else:
                    details['support'] = "未找到有效支撑位"
            else:
                details['support'] = "未找到有效支撑位"
                
            # 4. 获取MA交叉信号
            try:
                # 检查MA数据缓存是否需要更新
                if await self._should_update_indicator('ma_data'):
                    short_ma, long_ma = await self.trader.get_ma_data(short_period=20, long_period=50)
                    # 更新缓存
                    self.indicator_cache['ma_data'] = {
                        'value': (short_ma, long_ma),
                        'last_update': time.time()
                    }
                else:
                    # 使用缓存数据
                    short_ma, long_ma = self.indicator_cache['ma_data']['value']
                
                if short_ma and long_ma:
                    if short_ma > long_ma:  # 金叉形态，短期均线在长期均线之上
                        details['ma_cross'] = f"金叉形态，MA20 ({short_ma:.1f}) > MA50 ({long_ma:.1f})"
                        if current_price < short_ma:  # 价格回调至短期均线，可能是买入机会
                            support_score += 10
                    else:  # 死叉形态，短期均线在长期均线之下
                        details['ma_cross'] = f"死叉形态，MA20 ({short_ma:.1f}) < MA50 ({long_ma:.1f})"
                else:
                    details['ma_cross'] = "无法获取MA数据"
            except Exception as e:
                self.logger.error(f"计算MA数据失败: {e}")
                details['ma_cross'] = "计算MA数据失败"
                
            # 5. 获取MACD信号
            try:
                # 检查MACD数据缓存是否需要更新
                if await self._should_update_indicator('macd_data'):
                    macd_line, signal_line = await self.trader.get_macd_data()
                    # 更新缓存
                    self.indicator_cache['macd_data'] = {
                        'value': (macd_line, signal_line),
                        'last_update': time.time()
                    }
                else:
                    # 使用缓存数据
                    macd_line, signal_line = self.indicator_cache['macd_data']['value']
                
                if macd_line is not None and signal_line is not None:
                    if macd_line > signal_line:  # MACD金叉，买入信号
                        support_score += 10
                        details['macd'] = f"MACD金叉 ({macd_line:.4f} > {signal_line:.4f})"
                    else:
                        details['macd'] = f"MACD死叉 ({macd_line:.4f} < {signal_line:.4f})"
                else:
                    details['macd'] = "无法获取MACD数据"
            except Exception as e:
                self.logger.error(f"计算MACD数据失败: {e}")
                details['macd'] = "计算MACD数据失败"
                
            # 6. 价格分位检查
            try:
                # 检查价格分位数缓存是否需要更新
                if await self._should_update_indicator('price_percentile'):
                    price_percentile = await self.trader._get_price_percentile()
                    # 更新缓存
                    self.indicator_cache['price_percentile'] = {
                        'value': price_percentile,
                        'last_update': time.time()
                    }
                else:
                    # 使用缓存数据
                    price_percentile = self.indicator_cache['price_percentile']['value']
                
                if price_percentile <= 0.3:  # 价格处于30%分位数以下，较好的买入位置
                    support_score += 15
                    details['price_percentile'] = f"价格处于低位 ({price_percentile:.2f})"
                elif price_percentile <= 0.5:  # 价格处于中位数以下
                    support_score += 5
                    details['price_percentile'] = f"价格处于中低位 ({price_percentile:.2f})"
                else:
                    details['price_percentile'] = f"价格处于高位 ({price_percentile:.2f})"
            except Exception as e:
                self.logger.error(f"计算价格分位失败: {e}")
                details['price_percentile'] = "价格分位计算失败"
                
            # 7. 短期趋势检查
            short_term_trend = 0
            
            try:
                # 检查短期趋势缓存是否需要更新
                if await self._should_update_indicator('short_trend'):
                    # 使用已缓存的K线数据
                    klines = await self._get_klines_data('1h', 24, force_update=False)
                    
                    if klines and len(klines) >= 24:
                        # 提取收盘价
                        closes = [float(kline[4]) for kline in klines]
                        
                        # 简单计算短期趋势(24小时价格变化)
                        price_change = (closes[-1] - closes[0]) / closes[0]
                        
                        # 更新缓存
                        self.indicator_cache['short_trend'] = {
                            'value': (price_change, closes[-1]),
                            'last_update': time.time()
                        }
                    else:
                        price_change = 0
                else:
                    # 使用缓存数据
                    price_change, _ = self.indicator_cache['short_trend']['value']
                
                if price_change < -0.05:  # 短期下跌超过5%
                    details['short_trend'] = f"短期下跌趋势 ({price_change:.2%})"
                elif price_change > 0.05:  # 短期上涨超过5%
                    details['short_trend'] = f"短期上涨趋势 ({price_change:.2%})"
                    short_term_trend = -10  # 短期上涨过快，可能不是好的买入时机
                else:
                    details['short_trend'] = f"短期横盘整理 ({price_change:.2%})"
            except Exception as e:
                self.logger.error(f"计算短期趋势失败: {e}")
                details['short_trend'] = "短期趋势计算失败"
                
            # 综合计算最终得分 (0-100)
            final_score = min(
                100,  # 最高分不超过100
                score + rsi_score + bb_score + support_score + short_term_trend
            )
            
            # 确定目标买入价格
            target_price = current_price  # 默认目标价为当前价
            
            # 如果有支撑位，可以设置略高于支撑位的价格作为目标价
            if support_levels:
                nearest_support = max([s for s in support_levels if s < current_price * 0.99], default=None)
                if nearest_support:
                    # 设置目标价为当前价和最近支撑位的加权平均
                    target_price = (current_price * 0.3 + nearest_support * 0.7)
                    
            # 如果有布林带下轨，也可以考虑
            if bb_4h[2]:
                if bb_4h[2] < current_price and bb_4h[2] > target_price * 0.97:
                    # 如果布林带下轨接近目标价，可以采用平均值
                    target_price = (target_price + bb_4h[2]) / 2
            
            # 如果价格已经很低，就直接使用当前价
            if final_score >= 80:
                target_price = current_price
            
            # 添加评分历史记录
            current_time = time.time()
            self.recent_scores.append((current_time, final_score))
            
            # 只保留最近30分钟的评分历史
            self.recent_scores = [x for x in self.recent_scores if current_time - x[0] < 1800]
            
            # 如果有近期评分，对当前评分进行平滑处理
            if len(self.recent_scores) > 1:
                # 计算最后一次评估的时间和评分
                last_eval_time, last_score = self.recent_scores[-2]
                time_diff = current_time - last_eval_time
                
                # 根据时间间隔确定平滑程度
                if time_diff < 60:  # 1分钟内
                    # 短时间内的评分波动可能不可靠，保留70%的前值
                    smoothed_score = final_score * 0.3 + last_score * 0.7
                    self.logger.debug(f"评分平滑(1分钟内): 原始={final_score:.1f}, 上次={last_score:.1f}, 平滑后={smoothed_score:.1f}")
                    final_score = smoothed_score
                elif time_diff < 300:  # 5分钟内
                    # 中等时间间隔，保留50%的前值
                    smoothed_score = final_score * 0.5 + last_score * 0.5
                    self.logger.debug(f"评分平滑(5分钟内): 原始={final_score:.1f}, 上次={last_score:.1f}, 平滑后={smoothed_score:.1f}")
                    final_score = smoothed_score
                elif time_diff < 600:  # 10分钟内
                    # 较长时间间隔，保留30%的前值
                    smoothed_score = final_score * 0.7 + last_score * 0.3
                    self.logger.debug(f"评分平滑(10分钟内): 原始={final_score:.1f}, 上次={last_score:.1f}, 平滑后={smoothed_score:.1f}")
                    final_score = smoothed_score
                
                # 如果平滑后的评分跨越了决策阈值，记录警告日志
                thresholds = [40, 60, 80]
                for threshold in thresholds:
                    if (last_score < threshold and final_score >= threshold) or (last_score >= threshold and final_score < threshold):
                        self.logger.warning(f"评分平滑后跨越决策阈值{threshold}: {last_score:.1f} -> {final_score:.1f}，可能影响决策一致性")
            
            self.logger.info(
                f"入场条件评估 | "
                f"得分: {final_score}/100 | "
                f"当前价: {current_price:.2f} | "
                f"目标价: {target_price:.2f}"
            )
            
            # 处理详情为字符串，方便打印
            details_str = " | ".join([f"{k}: {v}" for k, v in details.items()])
            self.logger.debug(f"入场详情: {details_str}")
            
            return final_score, details, target_price
            
        except Exception as e:
            self.logger.error(f"评估入场条件失败: {str(e)}")
            return 0, {'error': f'评估入场条件失败: {str(e)}'}, None

    async def _detect_extreme_market(self):
        """
        检测市场是否处于极端状态（如暴跌、暴涨、异常波动等）
        
        Returns:
            tuple: (is_extreme, extreme_type, extreme_level)
                is_extreme: 布尔值，表示是否处于极端状态
                extreme_type: 字符串，极端状态类型 ('crash', 'pump', 'volatile', None)
                extreme_level: 浮点数，极端程度 (0.0-1.0)
        """
        try:
            # 获取多个时间周期的K线数据
            klines_1h = await self._get_klines_data('1h', 24, force_update=False)
            klines_4h = await self._get_klines_data('4h', 12, force_update=False)
            
            if not klines_1h or not klines_4h:
                return False, None, 0
            
            # 计算短期价格变化
            closes_1h = np.array([float(kline[4]) for kline in klines_1h])
            price_change_24h = (closes_1h[-1] / closes_1h[0] - 1) if len(closes_1h) >= 24 else 0
            
            # 计算最近6小时价格变化
            if len(closes_1h) >= 6:
                price_change_6h = closes_1h[-1] / closes_1h[-6] - 1
            else:
                price_change_6h = 0
            
            # 计算波动率
            volatility = await self._calculate_market_volatility()
            
            # 检测闪崩
            if price_change_24h < -0.15 or price_change_6h < -0.08:
                extreme_level = min(abs(price_change_24h) * 2, 1.0)
                return True, 'crash', extreme_level
            
            # 检测暴涨
            if price_change_24h > 0.15 or price_change_6h > 0.08:
                extreme_level = min(abs(price_change_24h) * 2, 1.0)
                return True, 'pump', extreme_level
            
            # 检测异常波动
            if volatility > 0.05:  # 波动率超过5%
                extreme_level = min(volatility * 10, 1.0)
                return True, 'volatile', extreme_level
            
            return False, None, 0
            
        except Exception as e:
            self.logger.error(f"检测极端市场失败: {e}")
            return False, None, 0
    
    async def recover_min_position(self, current_position_ratio):
        """
        当仓位低于最小值时，执行底仓恢复买入操作
        
        Args:
            current_position_ratio: 当前仓位比例
        """
        try:
            # 使用状态锁保护共享状态
            async with self.state_lock:
                self.logger.info("开始执行底仓恢复操作...")
                
                # 获取最新的仓位比例，不依赖传入的参数值
                latest_position_ratio = await self._get_position_ratio()
                
                # 计算需要达到的目标仓位比例（最小仓位比例）
                target_ratio = self.trader.config.MIN_POSITION_RATIO
                
                # 如果当前仓位已经达到或超过目标，则不需要操作
                if latest_position_ratio >= target_ratio:
                    self.logger.info(f"当前仓位 {latest_position_ratio:.2%} 已达到或超过目标 {target_ratio:.2%}，无需恢复")
                    # 如果有待执行的买入任务，也取消它
                    if self.pending_recovery:
                        self.logger.info("取消待执行的买入任务，因为仓位已恢复")
                        self.pending_recovery = False
                    return True
                
                # 使用最新的仓位比例更新当前值，确保后续计算基于最新数据
                current_position_ratio = latest_position_ratio
                
                # 检测市场是否处于极端状态
                is_extreme, extreme_type, extreme_level = await self._detect_extreme_market()
                
                # 获取当前总资产价值
                total_assets = await self.trader._get_total_assets()
                
                # 计算当前仓位价值
                position_value = await self._get_position_value()
                
                # 计算需要买入的价值（目标仓位价值 - 当前仓位价值）
                target_position_value = total_assets * target_ratio
                buy_value_needed = target_position_value - position_value
                
                # 评估入场条件
                entry_score, entry_details, target_price = await self._evaluate_entry_conditions()
                current_price = await self._get_latest_price_with_retry()
                
                # 在极端市场状况下调整策略
                if is_extreme:
                    self.logger.warning(f"检测到极端市场状况: {extreme_type}, 程度: {extreme_level:.2f}")
                    
                    if extreme_type == 'crash':
                        # 在闪崩中采取更谨慎策略，根据崩盘严重程度调整
                        self.logger.info("市场闪崩中，采取分批买入策略")
                        
                        # 根据崩盘程度调整买入力度
                        if extreme_level > 0.7:  # 严重崩盘
                            # 仅买入计划的20%，剩余等待
                            buy_value_adjustment = 0.2
                            self.logger.info(f"严重崩盘，仅买入计划的{buy_value_adjustment*100:.0f}%")
                        elif extreme_level > 0.4:  # 中度崩盘
                            # 买入计划的40%
                            buy_value_adjustment = 0.4
                            self.logger.info(f"中度崩盘，买入计划的{buy_value_adjustment*100:.0f}%")
                        else:  # 轻度崩盘
                            # 买入计划的60%
                            buy_value_adjustment = 0.6
                            self.logger.info(f"轻度崩盘，买入计划的{buy_value_adjustment*100:.0f}%")
                        
                        # 调整买入金额
                        buy_value_needed = buy_value_needed * buy_value_adjustment
                        
                        # 强制设置待执行买入任务，在更低价格买入剩余部分
                        self.pending_recovery = True
                        self.pending_recovery_amount = (target_position_value - position_value) * (1 - buy_value_adjustment)
                        # 设置更低的目标价格
                        lower_target = current_price * (1 - extreme_level * 0.05)  # 根据崩盘程度设置更低目标价
                        self.pending_recovery_target_price = min(target_price, lower_target)
                        self.pending_recovery_price_expiry = time.time() + 86400 * 2  # 延长到48小时
                        
                        self.logger.info(
                            f"设置等待买入任务: 金额 {self.pending_recovery_amount:.2f} USDT, "
                            f"目标价 {self.pending_recovery_target_price:.2f}, "
                            f"有效期 48 小时"
                        )
                    
                    elif extreme_type == 'pump':
                        # 在暴涨行情中更加保守
                        self.logger.info("市场暴涨中，采取保守策略")
                        
                        # 暴涨可能是买入的不好时机，减小买入力度
                        if entry_score < 60:  # 评分不够高
                            self.logger.info(f"暴涨行情评分较低({entry_score})，暂缓买入")
                            
                            # 设置或更新待执行买入任务，等待回调
                            self.pending_recovery = True
                            self.pending_recovery_amount = buy_value_needed
                            
                            # 设置目标价格为当前价格的回调位
                            pullback_target = current_price * (1 - 0.03)  # 回调3%
                            self.pending_recovery_target_price = pullback_target
                            self.pending_recovery_price_expiry = time.time() + 86400  # 24小时有效期
                            
                            self.logger.info(
                                f"等待价格回调，设置目标价: {pullback_target:.2f} (当前价的97%)"
                            )
                            return True  # 暂不买入，等待回调
                        else:
                            # 即使在上涨趋势中，如果技术指标极好，仍可少量买入
                            buy_value_needed = buy_value_needed * 0.3  # 只买入30%
                            self.logger.info(f"尽管市场暴涨，但评分较高({entry_score})，买入30%目标金额")
                    
                    elif extreme_type == 'volatile':
                        # 在高波动市场中分批买入
                        self.logger.info("市场波动剧烈，采取分批买入策略")
                        
                        # 调整买入比例
                        buy_value_needed = buy_value_needed * 0.5  # 只买入50%
                        
                        # 设置待执行买入任务
                        self.pending_recovery = True
                        self.pending_recovery_amount = buy_value_needed
                        # 设置目标价格稍低于当前价格
                        self.pending_recovery_target_price = current_price * 0.98  # 当前价格的98%
                        self.pending_recovery_price_expiry = time.time() + 86400  # 24小时有效期
                
                # 根据评分决定行动方案（非极端市场或极端市场已调整）
                if not is_extreme:
                    if entry_score >= 80:  # 非常好的买入机会，立即全部买入
                        self.logger.info(f"入场得分: {entry_score}，非常好的买入机会，立即全部买入")
                        buy_ratio = 1.0  # 买入全部目标金额
                        buy_target_price = current_price  # 使用市价买入
                        buy_reason = f"技术指标共振，满足优质入场条件 (得分: {entry_score}/100)"
                    elif entry_score >= 60:  # 较好的买入机会，买入70%
                        self.logger.info(f"入场得分: {entry_score}，较好的买入机会，买入70%目标金额")
                        buy_ratio = 0.7  # 买入70%目标金额
                        buy_target_price = target_price  # 使用评估出的目标价
                        buy_reason = f"多数技术指标给出买入信号 (得分: {entry_score}/100)"
                    elif entry_score >= 40:  # 一般买入机会，买入50%
                        self.logger.info(f"入场得分: {entry_score}，一般买入机会，买入50%目标金额")
                        buy_ratio = 0.5  # 买入50%目标金额
                        buy_target_price = target_price  # 使用评估出的目标价
                        buy_reason = f"部分技术指标给出买入信号 (得分: {entry_score}/100)"
                    else:  # 不是好的买入机会，仅买入30%，或等待更好时机
                        if self.pending_recovery:  # 已经有等待中的买入任务
                            self.logger.info(f"入场得分较低: {entry_score}，且已有等待中的买入任务，放弃本次买入")
                            # 更新目标价格，如果新的更低
                            if target_price < self.pending_recovery_target_price:
                                self.pending_recovery_target_price = target_price
                                self.pending_recovery_price_expiry = time.time() + 86400  # 24小时有效期
                                self.logger.info(f"更新等待买入目标价: {target_price:.2f}，有效期至 {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.pending_recovery_price_expiry))}")
                            return True
                        else:
                            self.logger.info(f"入场得分较低: {entry_score}，买入30%目标金额，并设置等待任务")
                            buy_ratio = 0.3  # 仅买入30%目标金额
                            buy_target_price = current_price  # 使用当前价格
                            buy_reason = f"技术指标不支持大幅买入 (得分: {entry_score}/100)"
                            
                            # 设置等待任务
                            self.pending_recovery = True
                            self.pending_recovery_amount = buy_value_needed * (1 - buy_ratio)  # 剩余待买入金额
                            self.pending_recovery_target_price = target_price
                            self.pending_recovery_price_expiry = time.time() + 86400  # 24小时有效期
                            self.logger.info(f"设置等待买入任务: 金额 {self.pending_recovery_amount:.2f} USDT，目标价 {target_price:.2f}，有效期至 {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.pending_recovery_price_expiry))}")
                
                    # 计算本次实际买入金额
                    actual_buy_value = buy_value_needed * buy_ratio
                else:
                    # 极端市场已在前面计算好买入金额
                    actual_buy_value = buy_value_needed
                    buy_reason = f"极端市场({extreme_type})调整策略，程度:{extreme_level:.2f} (评分: {entry_score}/100)"
                
                # 获取当前价格
                if not current_price or current_price <= 0:
                    self.logger.error("获取当前价格失败，无法执行底仓恢复")
                    return False
                
                # 判断价格合适性
                price_acceptable = False
                
                # 根据评分决定价格接受范围
                if entry_score >= 80:  # 高评分：无论价格如何
                    price_acceptable = True
                    self.logger.info(f"入场得分高({entry_score})，无论价格如何都接受市价买入")
                elif entry_score >= 60:  # 中评分：当前价格在目标价格1.5%范围内
                    self.logger.info(f"入场得分中({entry_score})")
                    if current_price <= target_price * 1.015:
                        price_acceptable = True
                        self.logger.info(f"当前价格({current_price:.2f})在目标价格({target_price:.2f})的1.5%范围内，接受市价买入")
                elif entry_score >= 40:  # 低评分：当前价格在目标价格1%范围内
                    self.logger.info(f"入场得分低({entry_score})")
                    if current_price <= target_price * 1.01:
                        price_acceptable = True
                        self.logger.info(f"当前价格({current_price:.2f})在目标价格({target_price:.2f})的1%范围内，接受市价买入")
                else:  # 极低评分：当前价格在目标价格0.5%范围内
                    self.logger.info(f"入场得分极低({entry_score}),等待机会")
                    # 动态价格偏差容忍度
                    volatility = await self._calculate_market_volatility()
                    price_deviation = min(0.005 + (volatility * 0.05), 0.02)  # 基础0.5% + 波动贡献，最高2%
                    
                    if current_price <= target_price * (1 + price_deviation):
                        price_acceptable = True
                        self.logger.info(f"当前价格({current_price:.2f})在目标价格({target_price:.2f})的{price_deviation:.2%}范围内，接受市价买入")
                        
                # 如果价格不可接受，则设置或更新pending_recovery任务并跳过当前买入
                if not price_acceptable:
                    # 设置或更新pending_recovery任务
                    self.pending_recovery = True
                    self.pending_recovery_amount = actual_buy_value  # 待买入金额
                    self.pending_recovery_target_price = target_price
                    self.pending_recovery_price_expiry = time.time() + 86400  # 24小时有效期
                    self.logger.info(f"价格不合适，跳过当前买入，设置等待任务: 金额 {self.pending_recovery_amount:.2f} USDT，目标价 {target_price:.2f}，有效期至 {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.pending_recovery_price_expiry))}")
                    return True  # 执行成功（虽然没有买入，但系统正常运行）
                
                # 计算需要买入的数量 (只基于市价)
                buy_amount = actual_buy_value / current_price
                
                # 确保买入数量大于最小交易数量
                min_trade_amount = self.trader.config.MIN_TRADE_AMOUNT / current_price
                if buy_amount < min_trade_amount:
                    self.logger.info(f"计算的买入数量 {buy_amount:.8f} 小于最小交易数量 {min_trade_amount:.8f}，调整为最小交易数量")
                    buy_amount = min_trade_amount
                
                # 调整精度
                if hasattr(self.trader, '_adjust_amount_precision') and callable(self.trader._adjust_amount_precision):
                    buy_amount = self.trader._adjust_amount_precision(buy_amount)
                
                # 检查USDT余额是否足够 (只基于市价)
                usdt_needed = buy_amount * current_price
                usdt_available = await self.trader.get_available_balance('USDT')
                
                if usdt_available < usdt_needed:
                    self.logger.info(f"USDT余额不足，需要 {usdt_needed:.2f}，可用 {usdt_available:.2f}，尝试从理财赎回")
                    
                    # 使用交易者的资金转移方法
                    if hasattr(self.trader, '_pre_transfer_funds'):
                        try:
                            await self.trader._pre_transfer_funds(current_price)
                            # 重新检查余额
                            usdt_available = await self.trader.get_available_balance('USDT')
                            if usdt_available < usdt_needed:
                                self.logger.warning(f"即使赎回后，USDT余额仍不足，可用 {usdt_available:.2f}")
                                return False
                                
                            # 赎回成功后，重新评估市场状况
                            new_entry_score, _, new_target_price = await self._evaluate_entry_conditions()
                            
                            # 如果市场状况显著恶化，调整买入策略
                            if new_entry_score < entry_score * 0.8:  # 评分下降超过20%
                                self.logger.warning(f"资金赎回期间市场状况变化，评分从 {entry_score} 降至 {new_entry_score}，调整策略")
                                
                                # 如果评分大幅下降，减少买入数量或取消买入
                                if new_entry_score < 30:
                                    self.logger.info("评分显著降低，取消当前买入，设置等待任务")
                                    
                                    # 设置等待任务
                                    self.pending_recovery = True
                                    self.pending_recovery_amount = actual_buy_value
                                    self.pending_recovery_target_price = new_target_price
                                    self.pending_recovery_price_expiry = time.time() + 86400  # 24小时有效期
                                    return True
                                else:
                                    # 减少买入量到原计划的50%
                                    self.logger.info("评分降低，减少买入量到原计划的50%")
                                    buy_amount = buy_amount * 0.5
                                    
                                    # 设置等待任务买入剩余部分
                                    self.pending_recovery = True
                                    self.pending_recovery_amount = actual_buy_value * 0.5
                                    self.pending_recovery_target_price = new_target_price
                                    self.pending_recovery_price_expiry = time.time() + 86400  # 24小时有效期
                            
                        except Exception as e:
                            self.logger.error(f"从理财赎回资金失败: {e}")
                            return False
                    else:
                        self.logger.warning("无法从理财赎回资金，trader没有_pre_transfer_funds方法")
                        return False
                
                # 执行买入
                try:
                    # 准备通知消息前缀
                    if is_extreme:
                        notification_prefix = f"极端市场底仓恢复买入 ({extreme_type})\n"
                    else:
                        notification_prefix = f"底仓恢复买入 ({buy_ratio*100:.0f}%)\n"
                    
                    # 使用市价单买入
                    self.logger.info(f"执行底仓恢复市价买入: {buy_amount:.8f} 个币种，预计花费 {usdt_needed:.2f} USDT")
                    
                    # 使用交易者的交易方法执行买入
                    if hasattr(self.trader, '_execute_trade') and callable(self.trader._execute_trade):
                        order = await self.trader._execute_trade('buy', current_price, buy_amount)
                        self.logger.info(f"底仓恢复买入成功，订单ID: {order.get('id', 'N/A')}")
                        
                        # 更新交易记录
                        if hasattr(self.trader, 'order_tracker'):
                            trade_info = {
                                'timestamp': time.time(),
                                'strategy': 'BASE_RECOVERY',  # 标记为底仓恢复策略
                                'side': 'buy',
                                'price': float(order.get('average', current_price)),
                                'amount': float(order.get('filled', buy_amount)),
                                'order_id': order.get('id'),
                                'note': buy_reason  # 添加买入理由
                            }
                            self.trader.order_tracker.add_trade(trade_info)
                    else:
                        # 直接使用交易所客户端执行买入
                        order = await self.trader.exchange.create_market_order(
                            symbol=self.trader.symbol,
                            side='buy',
                            amount=buy_amount
                        )
                        self.logger.info(f"底仓恢复买入成功，订单ID: {order.get('id', 'N/A')}")
                    
                    # 计算实际成交价格和数量
                    filled_price = float(order.get('average', current_price))
                    filled_amount = float(order.get('filled', buy_amount))
                    filled_cost = filled_price * filled_amount
                    
                    # 更新最后已知价格
                    self.last_known_price = filled_price
                    
                    # 市价单的通知消息
                    notification_message = (
                        f"{notification_prefix}执行成功\n"
                        f"买入数量: {filled_amount:.8f}\n"
                        f"成交价格: {filled_price:.2f}\n"
                        f"花费USDT: {filled_cost:.2f}\n"
                        f"买入理由: {buy_reason}\n"
                        f"技术详情: {', '.join([f'{k}: {v}' for k, v in entry_details.items() if k in ['rsi_1d', 'bb_1d', 'support']])}"
                    )
                    
                    # 记录底仓恢复状态
                    self.last_recovery_time = time.time()
                    self.recovery_executed = True
                    
                    # 发送通知
                    if hasattr(self.trader, 'send_notification') and callable(self.trader.send_notification):
                        await self.trader.send_notification(notification_message)
                    else:
                        # 尝试使用已知的通知方法
                        try:
                            from helpers import send_pushplus_message, send_ntfy_message
                            send_pushplus_message(notification_message, "底仓恢复通知")
                            send_ntfy_message(
                                content=notification_message,
                                title="底仓恢复买入",
                                priority="high",
                                tags="chart_with_upwards_trend"
                            )
                        except Exception as e:
                            self.logger.error(f"发送底仓恢复通知失败: {e}")
                    
                    # 底仓恢复成功后，设置冷却期，避免短时间内重复触发
                    self.recovery_cooldown_until = time.time() + 3600  # 1小时冷却期
                    
                    # 执行完买入后，重新检查当前仓位是否已经达到或超过目标
                    new_position_ratio = await self._get_position_ratio()
                    if new_position_ratio >= target_ratio:
                        self.logger.info(f"买入后仓位 {new_position_ratio:.2%} 已达到目标 {target_ratio:.2%}，取消待执行买入任务")
                        self.pending_recovery = False
                    
                    return True  # 表示底仓恢复操作成功执行
                except Exception as e:
                    self.logger.error(f"执行底仓恢复买入失败: {e}")
                    return False  # 表示底仓恢复操作失败
                    
        except Exception as e:
            self.logger.error(f"底仓恢复操作失败: {str(e)}")
            return False  # 表示底仓恢复操作失败

    async def _get_position_value(self):
        balance = await self.trader.exchange.fetch_balance()
        funding_balance = await self.trader.exchange.fetch_funding_balance()
        if not self.trader.symbol_info:
            self.trader.trade_log.error("交易对信息未初始化")
            return 0
        base_amount = (
            float(balance.get('free', {}).get(self.trader.symbol_info['base'], 0)) +
            float(funding_balance.get(self.trader.symbol_info['base'], 0))
        )
        current_price = await self.trader._get_latest_price()
        return base_amount * current_price

    async def _get_position_ratio(self):
        """获取当前仓位占总资产比例"""
        try:
            position_value = await self._get_position_value()
            balance = await self.trader.exchange.fetch_balance()
            funding_balance = await self.trader.exchange.fetch_funding_balance()
            
            usdt_balance = (
                float(balance.get('free', {}).get('USDT', 0)) +
                float(funding_balance.get('USDT', 0))
            )
            
            total_assets = position_value + usdt_balance
            if total_assets == 0:
                return 0
                
            ratio = position_value / total_assets
            
            # 记录仓位比例历史
            current_time = time.time()
            self.position_ratio_history.append((current_time, ratio))
            
            # 只保留最近10分钟的历史
            self.position_ratio_history = [x for x in self.position_ratio_history 
                                          if current_time - x[0] < 600]
            
            # 计算时间加权平均比例，减轻短期价格波动的影响
            if len(self.position_ratio_history) > 1:
                # 计算时间权重，更近的数据权重更高
                time_weights = [(current_time - x[0]) / 600 for x in self.position_ratio_history]
                inv_weights = [1 - w for w in time_weights]  # 反转权重，使更近的数据权重更高
                total_weight = sum(inv_weights) if sum(inv_weights) > 0 else 1
                normalized_weights = [w / total_weight for w in inv_weights]
                
                # 计算加权平均
                weighted_ratio = sum(w * r for w, (_, r) in zip(normalized_weights, self.position_ratio_history))
                
                # 融合当前比例和加权历史比例
                smoothed_ratio = (ratio * 0.7) + (weighted_ratio * 0.3)
                
                # 如果平滑后的比例与当前比例相差超过2%，记录日志
                if abs(smoothed_ratio - ratio) > 0.02:
                    self.logger.debug(
                        f"仓位比例平滑: 原始={ratio:.2%}, 平滑后={smoothed_ratio:.2%}, "
                        f"差异={(smoothed_ratio-ratio):.2%}"
                    )
                
                ratio = smoothed_ratio
            
            self.logger.debug(
                f"仓位计算 | "
                f"BNB价值: {position_value:.2f} USDT | "
                f"USDT余额: {usdt_balance:.2f} | "
                f"总资产: {total_assets:.2f} | "
                f"仓位比例: {ratio:.2%}"
            )
            return ratio
        except Exception as e:
            self.logger.error(f"计算仓位比例失败: {str(e)}")
            return 0

    async def _check_pending_recovery(self):
        """
        检查并执行待处理的底仓恢复买入任务
        """
        try:
            # 使用状态锁保护共享状态
            async with self.state_lock:
                # 检查任务是否过期
                current_time = time.time()
                if self.pending_recovery and current_time > self.pending_recovery_price_expiry:
                    self.logger.info(f"待执行买入任务已过期，清理并重新评估")
                    self.pending_recovery = False
                    # 在这里可以选择是否重新评估市场并设置新任务
                    # 暂时不自动设置新任务，等待下次底仓检查触发
                    return False
            
                # 先检查当前仓位是否已经恢复
                current_position_ratio = await self._get_position_ratio()
                target_ratio = self.trader.config.MIN_POSITION_RATIO
                
                # 如果仓位已经恢复，取消待执行任务
                if current_position_ratio >= target_ratio:
                    self.logger.info(
                        f"待执行买入任务取消 | "
                        f"当前仓位 {current_position_ratio:.2%} 已达到目标 {target_ratio:.2%}"
                    )
                    self.pending_recovery = False
                    return True
                    
                # 检查目标价格是否已经达到
                current_price = await self._get_latest_price_with_retry()
                
                # 动态计算价格偏差容忍度
                volatility = await self._calculate_market_volatility()
                price_deviation = min(0.005 + (volatility * 0.05), 0.02)  # 基础0.5% + 波动贡献，最高2%
                
                if current_price <= self.pending_recovery_target_price * (1 + price_deviation):
                    self.logger.info(
                        f"待执行买入任务价格条件满足 | "
                        f"目标价: {self.pending_recovery_target_price:.2f} | "
                        f"当前价: {current_price:.2f} | "
                        f"价格偏差容忍度: {price_deviation:.2%} | "
                        f"买入金额: {self.pending_recovery_amount:.2f} USDT"
                    )
                    
                    # 新增：重新评估当前市场状况，不使用缓存
                    self.logger.info("重新评估当前市场状况，确认是否适合买入...")
                    # 批量预获取K线数据，强制更新以获取最新数据
                    await self._get_klines_data('4h', 50, force_update=True)  # 用于RSI和布林带
                    await self._get_klines_data('1d', 90, force_update=True)  # 用于日线指标和支撑位
                    await self._get_klines_data('1h', 24, force_update=True)  # 用于短期趋势
                    
                    # 重新评估入场条件
                    entry_score, entry_details, new_target_price = await self._evaluate_entry_conditions()
                    
                    # 根据最新评分决定是否执行买入
                    if entry_score >= 80:  # 非常好的买入机会，全部买入
                        self.logger.info(f"重新评估得分: {entry_score}，市场条件非常好，执行全部买入")
                        # 继续执行原计划的买入
                        buy_amount = self.pending_recovery_amount / current_price
                        buy_reason = f"重新评估后市场状况极佳 (得分: {entry_score}/100)"
                    elif entry_score >= 60:  # 较好的买入机会，执行原计划
                        self.logger.info(f"重新评估得分: {entry_score}，市场条件良好，执行预定买入")
                        # 继续执行原计划的买入
                        buy_amount = self.pending_recovery_amount / current_price
                        buy_reason = f"重新评估后市场状况良好 (得分: {entry_score}/100)"
                    elif entry_score >= 40:  # 一般买入机会，减少买入量
                        self.logger.info(f"重新评估得分: {entry_score}，市场条件一般，减少买入量")
                        # 减少买入量至原计划的50%
                        buy_amount = (self.pending_recovery_amount * 0.5) / current_price
                        # 更新待执行任务，剩余部分在更好的价格买入
                        self.pending_recovery = True
                        self.pending_recovery_amount = self.pending_recovery_amount * 0.5  # 保留50%未买入
                        if new_target_price < self.pending_recovery_target_price:
                            self.pending_recovery_target_price = new_target_price
                            self.logger.info(f"更新目标价格为: {new_target_price:.2f}")
                        self.pending_recovery_price_expiry = time.time() + 86400  # 延长24小时
                        buy_reason = f"重新评估后市场状况一般，减少买入量 (得分: {entry_score}/100)"
                    else:  # 不好的买入机会，取消当前买入，更新目标价格
                        self.logger.info(f"重新评估得分: {entry_score}，市场条件不佳，取消当前买入并更新目标价格")
                        # 更新目标价格并延长等待时间
                        if new_target_price < self.pending_recovery_target_price:
                            self.pending_recovery_target_price = new_target_price
                            self.pending_recovery_price_expiry = time.time() + 86400  # 延长24小时
                            self.logger.info(
                                f"取消当前买入并更新目标价格: {new_target_price:.2f}，"
                                f"有效期至: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.pending_recovery_price_expiry))}"
                            )
                        return False
                    
                    # 确保买入数量大于最小交易数量
                    min_trade_amount = self.trader.config.MIN_TRADE_AMOUNT / current_price
                    if buy_amount < min_trade_amount:
                        self.logger.info(f"计算的买入数量 {buy_amount:.8f} 小于最小交易数量 {min_trade_amount:.8f}，调整为最小交易数量")
                        buy_amount = min_trade_amount
                    
                    # 调整精度
                    if hasattr(self.trader, '_adjust_amount_precision') and callable(self.trader._adjust_amount_precision):
                        buy_amount = self.trader._adjust_amount_precision(buy_amount)
                    
                    # 检查USDT余额是否足够
                    usdt_needed = buy_amount * current_price
                    usdt_available = await self.trader.get_available_balance('USDT')
                    
                    if usdt_available < usdt_needed:
                        self.logger.info(f"USDT余额不足，需要 {usdt_needed:.2f}，可用 {usdt_available:.2f}，暂不执行待定买入")
                        return False
                    
                    # 执行买入
                    try:
                        # 添加技术指标详情到买入理由
                        tech_details = ", ".join([f"{k}: {v}" for k, v in entry_details.items() if k in ['rsi_1d', 'bb_1d', 'support', 'short_trend']])
                        buy_reason = f"{buy_reason}\n技术详情: {tech_details}"
                        
                        # 使用交易者的交易方法执行买入
                        if hasattr(self.trader, '_execute_trade') and callable(self.trader._execute_trade):
                            order = await self.trader._execute_trade('buy', current_price, buy_amount)
                            self.logger.info(f"待执行底仓恢复买入成功，订单ID: {order.get('id', 'N/A')}")
                            
                            # 更新交易记录
                            if hasattr(self.trader, 'order_tracker'):
                                trade_info = {
                                    'timestamp': time.time(),
                                    'strategy': 'BASE_RECOVERY_PENDING',  # 标记为待执行底仓恢复策略
                                    'side': 'buy',
                                    'price': float(order.get('average', current_price)),
                                    'amount': float(order.get('filled', buy_amount)),
                                    'order_id': order.get('id'),
                                    'note': f"待执行底仓恢复买入，重新评估得分: {entry_score}/100"
                                }
                                self.trader.order_tracker.add_trade(trade_info)
                        else:
                            # 直接使用交易所客户端执行买入
                            order = await self.trader.exchange.create_market_order(
                                symbol=self.trader.symbol,
                                side='buy',
                                amount=buy_amount
                            )
                            self.logger.info(f"待执行底仓恢复买入成功，订单ID: {order.get('id', 'N/A')}")
                        
                        # 计算实际成交价格和数量
                        filled_price = float(order.get('average', current_price))
                        filled_amount = float(order.get('filled', buy_amount))
                        filled_cost = filled_price * filled_amount
                        
                        # 更新最后已知价格
                        self.last_known_price = filled_price
                        
                        # 发送通知消息
                        notification_message = (
                            f"待执行底仓恢复买入成功\n"
                            f"买入数量: {filled_amount:.8f}\n"
                            f"成交价格: {filled_price:.2f}\n"
                            f"花费USDT: {filled_cost:.2f}\n"
                            f"买入理由: {buy_reason}\n"
                            f"原始目标价: {self.pending_recovery_target_price:.2f}\n"
                            f"最新评估: 得分 {entry_score}/100"
                        )
                        
                        # 发送通知
                        if hasattr(self.trader, 'send_notification') and callable(self.trader.send_notification):
                            await self.trader.send_notification(notification_message)
                        else:
                            # 尝试使用已知的通知方法
                            try:
                                from helpers import send_pushplus_message, send_ntfy_message
                                send_pushplus_message(notification_message, "底仓恢复通知")
                                send_ntfy_message(
                                    content=notification_message,
                                    title="待执行底仓恢复买入",
                                    priority="high",
                                    tags="chart_with_upwards_trend"
                                )
                            except Exception as e:
                                self.logger.error(f"发送待执行底仓恢复通知失败: {e}")
                        
                        # 如果买入后没有剩余待执行任务，重置状态
                        if entry_score >= 60 or not self.pending_recovery:
                            self.pending_recovery = False
                            self.logger.info("所有待执行买入任务已完成")
                        
                        return True
                        
                    except Exception as e:
                        self.logger.error(f"执行待定底仓恢复买入失败: {e}")
                        return False
                else:
                    self.logger.debug(
                        f"待执行买入任务等待中 | "
                        f"目标价: {self.pending_recovery_target_price:.2f} | "
                        f"当前价: {current_price:.2f} | "
                        f"价格偏差容忍度: {price_deviation:.2%} | "
                        f"有效期至: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.pending_recovery_price_expiry))}"
                    )
                    return False
        except Exception as e:
            self.logger.error(f"检查待执行买入任务失败: {str(e)}")
            return False

    async def check_market_sentiment(self):
        """检查市场情绪指标"""
        try:
            fear_greed = await self._get_fear_greed_index()
            if fear_greed < 20:  # 极度恐慌
                self.trader.config.RISK_FACTOR *= 0.5  # 降低风险系数
            elif fear_greed > 80:  # 极度贪婪
                self.trader.config.RISK_FACTOR *= 1.2  # 提高风险系数
        except Exception as e:
            self.logger.error(f"获取市场情绪失败: {str(e)}")

    async def _get_klines_data(self, timeframe, limit, force_update=False):
        """
        获取并缓存K线数据的辅助方法
        
        Args:
            timeframe: K线时间周期，如'1h', '4h', '1d'
            limit: 需要获取的K线数量
            force_update: 是否强制更新，忽略缓存
            
        Returns:
            list: K线数据列表
        """
        cache_key = f'klines_{timeframe}'
        
        # 检查缓存是否存在且未过期
        current_time = time.time()
        if (not force_update and 
            cache_key in self.indicator_cache and 
            self.indicator_cache[cache_key]['value'] is not None and 
            len(self.indicator_cache[cache_key]['value']) >= limit and
            current_time - self.indicator_cache[cache_key]['last_update'] < self.update_intervals.get(cache_key, 300)):
            
            # 返回缓存的数据，并确保数量正确
            return self.indicator_cache[cache_key]['value'][-limit:]
        
        try:
            # 获取比请求更多的数据，以备后续使用
            fetch_limit = max(limit * 2, 200)  # 获取足够的数据
            
            # 发送API请求获取K线数据
            klines = await self.trader.exchange.fetch_ohlcv(
                self.trader.symbol,
                timeframe=timeframe,
                limit=fetch_limit
            )
            
            if not klines or len(klines) < limit:
                self.logger.warning(f"获取K线数据失败: 获取的K线数据不足 ({len(klines) if klines else 0} < {limit})")
                # 如果有缓存，返回缓存
                if cache_key in self.indicator_cache and self.indicator_cache[cache_key]['value'] is not None:
                    return self.indicator_cache[cache_key]['value'][-limit:]
                return []
            
            # 更新缓存
            self.indicator_cache[cache_key] = {
                'value': klines,
                'last_update': current_time
            }
            
            return klines[-limit:]
            
        except Exception as e:
            self.logger.error(f"获取K线数据失败: {str(e)}")
            # 如果有缓存，返回缓存
            if cache_key in self.indicator_cache and self.indicator_cache[cache_key]['value'] is not None:
                return self.indicator_cache[cache_key]['value'][-limit:]
            return []
            
    async def _get_dynamic_update_interval(self, base_interval):
        """
        基于市场波动率动态调整指标更新间隔
        
        Args:
            base_interval: 基础更新间隔（秒）
            
        Returns:
            int: 调整后的更新间隔（秒）
        """
        try:
            # 获取当前波动率
            volatility = await self._calculate_market_volatility()
            
            # 根据波动率调整更新间隔
            if volatility > 0.05:  # 高波动性
                # 减少间隔，更频繁更新
                adjusted_interval = base_interval * 0.5
                return max(int(adjusted_interval), 60)  # 最短不低于60秒
            elif volatility < 0.01:  # 低波动性
                # 增加间隔，减少更新频率
                adjusted_interval = base_interval * 1.5
                return int(adjusted_interval)  # 延长50%
            
            # 中等波动性，使用基础间隔
            return base_interval
            
        except Exception as e:
            self.logger.error(f"计算动态更新间隔失败: {e}")
            return base_interval  # 出错时返回基础间隔
            
    async def _should_update_indicator(self, indicator_key):
        """
        检查指标是否需要更新
        
        Args:
            indicator_key: 指标缓存键名
            
        Returns:
            bool: 是否需要更新
        """
        current_time = time.time()
        
        # 获取基础更新间隔
        base_interval = self.update_intervals.get(indicator_key, 300)
        
        # 动态调整更新间隔
        dynamic_interval = await self._get_dynamic_update_interval(base_interval)
        
        # 检查是否首次获取或缓存已过期
        if (indicator_key not in self.indicator_cache or 
            self.indicator_cache[indicator_key]['value'] is None or 
            current_time - self.indicator_cache[indicator_key]['last_update'] >= dynamic_interval):
            
            # 如果是动态调整的间隔，记录日志
            if dynamic_interval != base_interval:
                self.logger.debug(f"指标 {indicator_key} 的更新间隔动态调整: {base_interval}秒 -> {dynamic_interval}秒")
                
            return True
            
        return False

    async def _get_latest_price_with_retry(self):
        """
        带重试机制的价格获取方法，增强API故障恢复能力
        
        Returns:
            float: 当前价格，如果获取失败则返回最后已知价格
        """
        retry_count = 0
        max_retries = 3
        
        while retry_count < max_retries:
            try:
                price = await self.trader._get_latest_price()
                if price and price > 0:
                    # 更新最后已知价格
                    self.last_known_price = price
                    return price
            except Exception as e:
                self.logger.error(f"获取价格失败 (尝试 {retry_count+1}/{max_retries}): {e}")
            
            retry_count += 1
            await asyncio.sleep(1)  # 短暂延迟后重试
        
        # 所有重试都失败，使用最后存储的价格
        if self.last_known_price:
            self.logger.warning(f"无法获取当前价格，使用最后已知价格: {self.last_known_price}")
            return self.last_known_price
        else:
            self.logger.error("无法获取当前价格，且没有历史价格记录")
            # 尝试从其他数据源获取价格...
            # 这里可以添加备用价格获取逻辑
            return None
    
    async def _calculate_market_volatility(self):
        """
        计算当前市场波动率，用于动态调整价格偏差和策略参数
        
        Returns:
            float: 市场波动率 (0.0-1.0)
        """
        try:
            # 获取1小时K线数据用于计算短期波动率
            klines_1h = await self._get_klines_data('1h', 24, force_update=False)
            
            if not klines_1h or len(klines_1h) < 6:  # 至少需要6根K线
                self.logger.warning("计算波动率失败: K线数据不足")
                return 0.02  # 返回默认中等波动率
            
            # 计算最近24根1小时K线的收盘价波动率（标准差/均值）
            closes = np.array([float(kline[4]) for kline in klines_1h])
            volatility = np.std(closes) / np.mean(closes)
            
            # 同时考虑价格范围波动率（高低点差异）
            high_prices = np.array([float(kline[2]) for kline in klines_1h])
            low_prices = np.array([float(kline[3]) for kline in klines_1h])
            price_range_volatility = (np.max(high_prices) - np.min(low_prices)) / np.mean(closes)
            
            # 将两种波动率结合
            combined_volatility = (volatility + price_range_volatility) / 2
            
            self.logger.debug(f"当前市场波动率: {combined_volatility:.4f}")
            return combined_volatility
            
        except Exception as e:
            self.logger.error(f"计算市场波动率失败: {e}")
            return 0.02  # 返回默认中等波动率 

    async def _get_fear_greed_index(self):
        """
        获取加密货币市场的恐慌贪婪指数
        恐慌贪婪指数是一个0-100的值，0表示极度恐慌，100表示极度贪婪
        
        Returns:
            int: 恐慌贪婪指数值(0-100)，如果获取失败则返回50(中性值)
        """
        indicator_key = 'fear_greed_index'
        
        # 检查缓存是否需要更新
        if not await self._should_update_indicator(indicator_key):
            return self.indicator_cache[indicator_key]['value']
            
        try:
            # 使用Alternative.me API获取恐慌贪婪指数
            async with aiohttp.ClientSession() as session:
                async with session.get('https://api.alternative.me/fng/?limit=1') as response:
                    if response.status == 200:
                        data = await response.json()
                        if data and 'data' in data and len(data['data']) > 0:
                            # 获取最新的恐慌贪婪指数值
                            fear_greed = int(data['data'][0]['value'])
                            classification = data['data'][0]['value_classification']
                            
                            self.logger.info(f"恐慌贪婪指数: {fear_greed}/100 ({classification})")
                            
                            # 更新缓存
                            self.indicator_cache[indicator_key] = {
                                'value': fear_greed,
                                'last_update': time.time()
                            }
                            
                            return fear_greed
                        else:
                            self.logger.warning("恐慌贪婪指数API返回数据格式异常")
                    else:
                        self.logger.warning(f"恐慌贪婪指数API请求失败: HTTP {response.status}")
            
            # 如果API请求失败，尝试备用API
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get('https://open-api-v4.coinglass.com/api/index/fear-greed-history', 
                                          headers={'coinglassSecret': 'your_api_key_if_needed'}) as response:
                        if response.status == 200:
                            data = await response.json()
                            if data and data.get('code') == '0' and 'data' in data and len(data['data']) > 0:
                                # CoinGlass API返回格式可能不同，需要适配
                                fear_greed_list = data['data'][0].get('data_list', [])
                                if fear_greed_list:
                                    fear_greed = int(fear_greed_list[0])
                                    self.logger.info(f"恐慌贪婪指数(备用API): {fear_greed}/100")
                                    
                                    # 更新缓存
                                    self.indicator_cache[indicator_key] = {
                                        'value': fear_greed,
                                        'last_update': time.time()
                                    }
                                    
                                    return fear_greed
            except Exception as e:
                self.logger.error(f"备用恐慌贪婪指数API请求失败: {e}")
            
            # 如果有缓存，返回缓存的值
            if indicator_key in self.indicator_cache and self.indicator_cache[indicator_key]['value'] is not None:
                self.logger.warning("无法获取最新恐慌贪婪指数，使用缓存值")
                return self.indicator_cache[indicator_key]['value']
                
            # 如果没有缓存，返回中性值50
            self.logger.warning("无法获取恐慌贪婪指数，使用默认中性值50")
            return 50
            
        except Exception as e:
            self.logger.error(f"获取恐慌贪婪指数失败: {str(e)}")
            
            # 如果有缓存，返回缓存的值
            if indicator_key in self.indicator_cache and self.indicator_cache[indicator_key]['value'] is not None:
                return self.indicator_cache[indicator_key]['value']
                
            # 如果没有缓存，返回中性值50
            return 50

    async def _safe_check_pending_recovery(self):
        """
        安全地检查并执行待处理的底仓恢复买入任务
        该方法包装了_check_pending_recovery，添加了额外的错误处理，
        确保即使出现异常也不会影响主流程
        """
        try:
            current_time = time.time()
            position_ratio = await self._get_position_ratio()
            # 如果仓位已经恢复到目标值但还有待执行的买入任务，取消它
            if position_ratio >= self.trader.config.MIN_POSITION_RATIO and self.pending_recovery:
                self.logger.info(f"仓位已恢复至 {position_ratio:.2%}，取消待执行的买入任务")
                self.pending_recovery = False

            if current_time < self.recovery_cooldown_until:
                cooldown_remaining = int(self.recovery_cooldown_until - current_time)
                self.logger.info(f"底仓恢复操作在冷却期内，还需等待 {cooldown_remaining} 秒")
                
                # 修改：将待定任务检查放入异步任务，不阻塞主流程
                if self.pending_recovery and current_time < self.pending_recovery_price_expiry:
                    if cooldown_remaining < 300:  # 冷却期剩余不到5分钟
                        # 创建异步任务检查并执行待定的买入任务，不等待其完成
                        await self._check_pending_recovery()
                        self.logger.info(f"已创建异步任务检查待执行买入任务")
                    else:
                        self.logger.info(f"待执行买入任务暂停执行，等待冷却期结束或接近结束")
            else:
                # 当仓位低于最小值时，尝试恢复底仓
                recovery_result = await self.recover_min_position(position_ratio)
                if recovery_result:
                    self.logger.info("底仓恢复操作已执行，等待下一次检查")
                else:
                    self.logger.warning("底仓恢复操作失败，将在下次检查时重试")
        
        except Exception as e:
            self.logger.error(f"安全检查待执行买入任务失败: {str(e)}")
            # 记录详细错误信息和堆栈跟踪
            import traceback
            self.logger.error(f"错误详情: {traceback.format_exc()}")
            return False
    
    async def _safe_check_pending_recovery_main(self):
        try:
            await asyncio.wait_for(self._safe_check_pending_recovery(), timeout=180)
        except asyncio.TimeoutError:
            self.logger.error("检查待执行买入任务超时，已取消操作")
            return False
        except Exception as e:
            self.logger.error(f"安全检查待执行买入任务失败: {str(e)}")
            # 记录详细错误信息和堆栈跟踪
            import traceback
            self.logger.error(f"错误详情: {traceback.format_exc()}")
            return False