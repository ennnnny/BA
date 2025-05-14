import logging
from config import MAX_POSITION_RATIO
import time
import numpy as np

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
    
    async def multi_layer_check(self):
        try:
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
                
                # 检查是否在冷却期内
                current_time = time.time()
                if current_time < self.recovery_cooldown_until:
                    cooldown_remaining = int(self.recovery_cooldown_until - current_time)
                    self.logger.info(f"底仓恢复操作在冷却期内，还需等待 {cooldown_remaining} 秒")
                    
                    # 新增：即使在冷却期内，也检查是否有待执行的买入任务且目标价格已达到
                    if self.pending_recovery and current_time < self.pending_recovery_price_expiry:
                        # 检查当前价格是否已达到目标价，是则执行待定的买入任务
                        await self._check_pending_recovery()
                else:
                    # 当仓位低于最小值时，尝试恢复底仓
                    recovery_result = await self.recover_min_position(position_ratio)
                    if recovery_result:
                        self.logger.info("底仓恢复操作已执行，等待下一次检查")
                    else:
                        self.logger.warning("底仓恢复操作失败，将在下次检查时重试")
                
                return True
            
            # 如果仓位已经恢复到目标值但还有待执行的买入任务，取消它
            if position_ratio >= self.trader.config.MIN_POSITION_RATIO and self.pending_recovery:
                self.logger.info(f"仓位已恢复至 {position_ratio:.2%}，取消待执行的买入任务")
                self.pending_recovery = False
            
            if position_ratio > self.trader.config.MAX_POSITION_RATIO:
                self.logger.warning(f"仓位超限 | 当前: {position_ratio:.2%}")
                return True
        except Exception as e:
            self.logger.error(f"风控检查失败: {str(e)}")
            return False

    async def _calculate_rsi(self, period=14, timeframe='4h'):
        """
        计算当前RSI值
        
        Args:
            period: RSI计算周期，默认14
            timeframe: K线时间周期，默认4小时
            
        Returns:
            float: RSI值(0-100)，如果计算失败则返回50
        """
        try:
            # 获取历史K线数据
            klines = await self.trader.exchange.fetch_ohlcv(
                self.trader.symbol,
                timeframe=timeframe,
                limit=period*2  # 获取足够的数据以计算RSI
            )
            
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
                return 100  # 避免除以零
                
            rs = avg_gain / avg_loss
            
            # 计算RSI
            rsi = 100 - (100 / (1 + rs))
            
            self.logger.debug(f"RSI({period}): {rsi:.2f}")
            return rsi
            
        except Exception as e:
            self.logger.error(f"计算RSI失败: {str(e)}")
            return 50  # 返回中性RSI值

    async def _calculate_bollinger_bands(self, period=20, deviation=2, timeframe='4h'):
        """
        计算布林带上中下轨
        
        Args:
            period: 移动平均周期，默认20
            deviation: 标准差倍数，默认2
            timeframe: K线时间周期，默认4小时
            
        Returns:
            tuple: (upper, middle, lower) 布林带上中下轨
                  如果计算失败则返回(None, None, None)
        """
        try:
            # 获取历史K线数据
            klines = await self.trader.exchange.fetch_ohlcv(
                self.trader.symbol,
                timeframe=timeframe,
                limit=period+10  # 获取足够的数据
            )
            
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
                f"布林带(周期={period}, 偏差={deviation}): "
                f"上轨={upper:.2f}, "
                f"中轨={middle:.2f}, "
                f"下轨={lower:.2f}, "
                f"当前价={current_price:.2f}, "
                f"位置={(current_price-lower)/(upper-lower) if upper != lower else 0.5:.2f}"
            )
            
            return upper, middle, lower
            
        except Exception as e:
            self.logger.error(f"计算布林带失败: {str(e)}")
            return None, None, None

    async def _get_support_levels(self, timeframe='1d', lookback=90):
        """
        获取市场关键支撑位
        
        Args:
            timeframe: K线时间周期，默认1天
            lookback: 回溯周期数，默认90
            
        Returns:
            list: 支撑位价格列表，按价格从低到高排序
        """
        try:
            # 获取历史K线数据
            klines = await self.trader.exchange.fetch_ohlcv(
                self.trader.symbol,
                timeframe=timeframe,
                limit=lookback  # 获取足够的数据用于分析
            )
            
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
            current_price = await self.trader._get_latest_price()
            if not current_price:
                return 0, {'error': '无法获取当前价格'}, None
                
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
                short_ma, long_ma = await self.trader.get_ma_data(short_period=20, long_period=50)
                
                if short_ma and long_ma:
                    if short_ma > long_ma:  # 金叉形态，短期均线在长期均线之上
                        details['ma_cross'] = f"金叉形态，MA20 ({short_ma:.1f}) > MA50 ({long_ma:.1f})"
                        if current_price < short_ma:  # 价格回调至短期均线，可能是买入机会
                            support_score += 10
                    else:  # 死叉形态，短期均线在长期均线之下
                        details['ma_cross'] = f"死叉形态，MA20 ({short_ma:.1f}) < MA50 ({long_ma:.1f})"
                else:
                    details['ma_cross'] = "无法获取MA数据"
            except Exception:
                details['ma_cross'] = "计算MA数据失败"
                
            # 5. 获取MACD信号
            try:
                macd_line, signal_line = await self.trader.get_macd_data()
                
                if macd_line is not None and signal_line is not None:
                    if macd_line > signal_line:  # MACD金叉，买入信号
                        support_score += 10
                        details['macd'] = f"MACD金叉 ({macd_line:.4f} > {signal_line:.4f})"
                    else:
                        details['macd'] = f"MACD死叉 ({macd_line:.4f} < {signal_line:.4f})"
                else:
                    details['macd'] = "无法获取MACD数据"
            except Exception:
                details['macd'] = "计算MACD数据失败"
                
            # 6. 价格分位检查
            try:
                price_percentile = await self.trader._get_price_percentile()
                
                if price_percentile <= 0.3:  # 价格处于30%分位数以下，较好的买入位置
                    support_score += 15
                    details['price_percentile'] = f"价格处于低位 ({price_percentile:.2f})"
                elif price_percentile <= 0.5:  # 价格处于中位数以下
                    support_score += 5
                    details['price_percentile'] = f"价格处于中低位 ({price_percentile:.2f})"
                else:
                    details['price_percentile'] = f"价格处于高位 ({price_percentile:.2f})"
            except Exception:
                details['price_percentile'] = "价格分位计算失败"
                
            # 7. 短期趋势检查
            short_term_trend = 0
            
            try:
                klines = await self.trader.exchange.fetch_ohlcv(
                    self.trader.symbol,
                    timeframe='1h',
                    limit=24  # 最近24小时
                )
                
                if klines and len(klines) >= 24:
                    # 提取收盘价
                    closes = [float(kline[4]) for kline in klines]
                    
                    # 简单计算短期趋势(24小时价格变化)
                    price_change = (closes[-1] - closes[0]) / closes[0]
                    
                    if price_change < -0.05:  # 短期下跌超过5%
                        details['short_trend'] = f"短期下跌趋势 ({price_change:.2%})"
                    elif price_change > 0.05:  # 短期上涨超过5%
                        details['short_trend'] = f"短期上涨趋势 ({price_change:.2%})"
                        short_term_trend = -10  # 短期上涨过快，可能不是好的买入时机
                    else:
                        details['short_trend'] = f"短期横盘整理 ({price_change:.2%})"
                else:
                    details['short_trend'] = "无法获取短期趋势数据"
            except Exception:
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

    async def recover_min_position(self, current_position_ratio):
        """
        当仓位低于最小值时，执行底仓恢复买入操作
        
        Args:
            current_position_ratio: 当前仓位比例
        """
        try:
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
            
            # 获取当前总资产价值
            total_assets = await self.trader._get_total_assets()
            
            # 计算当前仓位价值
            position_value = await self._get_position_value()
            
            # 计算需要买入的价值（目标仓位价值 - 当前仓位价值）
            target_position_value = total_assets * target_ratio
            buy_value_needed = target_position_value - position_value
            
            # 评估入场条件
            entry_score, entry_details, target_price = await self._evaluate_entry_conditions()
            current_price = await self.trader._get_latest_price()
            
            # 根据评分决定行动方案
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
                if current_price <= target_price * 1.005:
                    price_acceptable = True
                    self.logger.info(f"当前价格({current_price:.2f})在目标价格({target_price:.2f})的0.5%范围内，接受市价买入")
                    
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
                    except Exception as e:
                        self.logger.error(f"从理财赎回资金失败: {e}")
                        return False
                else:
                    self.logger.warning("无法从理财赎回资金，trader没有_pre_transfer_funds方法")
                    return False
            
            # 执行买入
            try:
                # 准备通知消息前缀
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
            current_price = await self.trader._get_latest_price()
            
            if current_price <= self.pending_recovery_target_price * 1.005:  # 允许0.5%的价格偏差
                self.logger.info(
                    f"待执行买入任务触发 | "
                    f"目标价: {self.pending_recovery_target_price:.2f} | "
                    f"当前价: {current_price:.2f} | "
                    f"买入金额: {self.pending_recovery_amount:.2f} USDT"
                )
                
                # 计算需要买入的数量
                buy_amount = self.pending_recovery_amount / current_price
                
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
                                'note': f"待执行底仓恢复买入，目标价: {self.pending_recovery_target_price:.2f}"
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
                    
                    # 发送通知消息
                    notification_message = (
                        f"待执行底仓恢复买入成功\n"
                        f"买入数量: {filled_amount:.8f}\n"
                        f"成交价格: {filled_price:.2f}\n"
                        f"花费USDT: {filled_cost:.2f}\n"
                        f"原始目标价: {self.pending_recovery_target_price:.2f}"
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
                    
                    # 重置待执行任务状态
                    self.pending_recovery = False
                    
                    return True
                    
                except Exception as e:
                    self.logger.error(f"执行待定底仓恢复买入失败: {e}")
                    return False
            else:
                self.logger.debug(
                    f"待执行买入任务等待中 | "
                    f"目标价: {self.pending_recovery_target_price:.2f} | "
                    f"当前价: {current_price:.2f} | "
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
            if fear_greed < 20:  # 极度恐惧
                self.trader.config.RISK_FACTOR *= 0.5  # 降低风险系数
            elif fear_greed > 80:  # 极度贪婪
                self.trader.config.RISK_FACTOR *= 1.2  # 提高风险系数
        except Exception as e:
            self.logger.error(f"获取市场情绪失败: {str(e)}") 