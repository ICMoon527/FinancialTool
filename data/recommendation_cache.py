import pandas as pd
from datetime import datetime, date
from typing import List, Dict, Any, Optional
from logger import log

try:
    from data.database import db_manager, StockRecommendation, HAS_SQLALCHEMY
except ImportError:
    HAS_SQLALCHEMY = False
    db_manager = None
    StockRecommendation = None


class RecommendationCache:
    def __init__(self):
        pass
    
    def _parse_date(self, date_val):
        if pd.isna(date_val) or date_val is None:
            return None
        if isinstance(date_val, datetime):
            return date_val.date()
        if isinstance(date_val, date):
            return date_val
        if isinstance(date_val, str):
            try:
                return datetime.strptime(date_val, '%Y%m%d').date()
            except:
                try:
                    return datetime.strptime(date_val, '%Y-%m-%d').date()
                except:
                    return None
        return None
    
    def has_valid_cache(self, horizon: str, recommendation_date: str = None) -> bool:
        """检查是否有有效的缓存"""
        if not HAS_SQLALCHEMY or not db_manager:
            return False
        
        if recommendation_date is None:
            recommendation_date = datetime.now().strftime('%Y%m%d')
        
        rec_date = self._parse_date(recommendation_date)
        if not rec_date:
            return False
        
        try:
            session = db_manager.get_session()
            try:
                count = session.query(StockRecommendation).filter(
                    StockRecommendation.horizon == horizon,
                    StockRecommendation.recommendation_date == rec_date
                ).count()
                return count > 0
            finally:
                session.close()
        except Exception as e:
            log.warning(f"检查缓存状态失败: {e}")
            return False
    
    def save_recommendations(self, recommendations: List[Dict], horizon: str, recommendation_date: str = None):
        """保存推荐结果到数据库"""
        if not HAS_SQLALCHEMY or not db_manager:
            log.warning("数据库不可用，无法保存推荐")
            return
        
        if recommendation_date is None:
            recommendation_date = datetime.now().strftime('%Y%m%d')
        
        rec_date = self._parse_date(recommendation_date)
        if not rec_date:
            log.warning("日期解析失败")
            return
        
        try:
            session = db_manager.get_session()
            try:
                session.query(StockRecommendation).filter(
                    StockRecommendation.horizon == horizon,
                    StockRecommendation.recommendation_date == rec_date
                ).delete()
                
                for idx, rec in enumerate(recommendations, 1):
                    analysis = rec.get('analysis', {})
                    record = StockRecommendation(
                        ts_code=rec.get('ts_code'),
                        name=rec.get('name'),
                        horizon=horizon,
                        score=float(rec.get('score', 0)),
                        rank=idx,
                        reason=rec.get('reason', ''),
                        target_price=float(analysis.get('target_price')) if analysis.get('target_price') is not None else None,
                        stop_loss=float(analysis.get('stop_loss')) if analysis.get('stop_loss') is not None else None,
                        holding_period=analysis.get('holding_period', ''),
                        entry_suggestion=analysis.get('entry_suggestion', ''),
                        risk_control=analysis.get('risk_control', ''),
                        recommendation_date=rec_date
                    )
                    session.add(record)
                
                session.commit()
                log.info(f"成功保存 {len(recommendations)} 条{horizon}推荐到缓存")
            except Exception as inner_e:
                session.rollback()
                raise inner_e
            finally:
                session.close()
        except Exception as e:
            log.error(f"保存推荐失败: {e}")
            import traceback
            traceback.print_exc()
    
    def load_recommendations(self, horizon: str, recommendation_date: str = None, top_n: int = 5) -> List[Dict]:
        """从数据库加载推荐结果"""
        if not HAS_SQLALCHEMY or not db_manager:
            log.warning("数据库不可用，无法加载推荐")
            return []
        
        if recommendation_date is None:
            recommendation_date = datetime.now().strftime('%Y%m%d')
        
        rec_date = self._parse_date(recommendation_date)
        if not rec_date:
            return []
        
        try:
            session = db_manager.get_session()
            try:
                query = session.query(StockRecommendation).filter(
                    StockRecommendation.horizon == horizon,
                    StockRecommendation.recommendation_date == rec_date
                ).order_by(StockRecommendation.rank).limit(top_n)
                
                results = query.all()
                
                recommendations = []
                for row in results:
                    rec = {
                        'ts_code': row.ts_code,
                        'name': row.name,
                        'score': row.score,
                        'reason': row.reason or '',
                        'analysis': {
                            'target_price': row.target_price,
                            'stop_loss': row.stop_loss,
                            'holding_period': row.holding_period or '',
                            'entry_suggestion': row.entry_suggestion or '',
                            'risk_control': row.risk_control or ''
                        }
                    }
                    
                    try:
                        from data.data_fetcher import data_fetcher
                        df = data_fetcher.load_stock_daily_from_db(row.ts_code)
                        if not df.empty and 'close' in df.columns:
                            rec['current_price'] = float(df['close'].iloc[-1])
                    except:
                        rec['current_price'] = float(row.target_price) if row.target_price is not None else 0
                    
                    recommendations.append(rec)
                
                log.info(f"从缓存加载 {len(recommendations)} 条{horizon}推荐")
                return recommendations
            finally:
                session.close()
        except Exception as e:
            log.error(f"加载推荐失败: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def clear_all_recommendations(self):
        """清除所有推荐缓存"""
        if not HAS_SQLALCHEMY or not db_manager:
            log.warning("数据库不可用，无法清除缓存")
            return
        
        try:
            session = db_manager.get_session()
            try:
                deleted_count = session.query(StockRecommendation).delete()
                session.commit()
                log.info(f"已清除 {deleted_count} 条推荐缓存")
                return deleted_count
            except Exception as inner_e:
                session.rollback()
                raise inner_e
            finally:
                session.close()
        except Exception as e:
            log.error(f"清除推荐缓存失败: {e}")
            import traceback
            traceback.print_exc()
            return 0
    
    def clear_recommendations_by_date(self, recommendation_date: str = None):
        """清除指定日期的推荐缓存"""
        if not HAS_SQLALCHEMY or not db_manager:
            log.warning("数据库不可用，无法清除缓存")
            return
        
        if recommendation_date is None:
            recommendation_date = datetime.now().strftime('%Y%m%d')
        
        rec_date = self._parse_date(recommendation_date)
        if not rec_date:
            return 0
        
        try:
            session = db_manager.get_session()
            try:
                deleted_count = session.query(StockRecommendation).filter(
                    StockRecommendation.recommendation_date == rec_date
                ).delete()
                session.commit()
                log.info(f"已清除 {recommendation_date} 的 {deleted_count} 条推荐缓存")
                return deleted_count
            except Exception as inner_e:
                session.rollback()
                raise inner_e
            finally:
                session.close()
        except Exception as e:
            log.error(f"清除推荐缓存失败: {e}")
            import traceback
            traceback.print_exc()
            return 0


recommendation_cache = RecommendationCache()
