"""
External Data Integrator for JARVIS.

Integrates external data sources to provide more informed suggestions and predictions.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import httpx
from pydantic import BaseModel

from core.cache import tool_cache
from tools.web.search import WebSearchTool

log = logging.getLogger(__name__)


class DataSourceType(Enum):
    """Types of external data sources"""

    WEATHER = "weather"
    NEWS = "news"
    STOCK_MARKET = "stock_market"
    SOCIAL_MEDIA = "social_media"
    CALENDAR = "calendar"
    EMAIL = "email"
    TRAFFIC = "traffic"
    SPORTS = "sports"
    ENTERTAINMENT = "entertainment"
    HEALTH = "health"
    FINANCIAL = "financial"


class DataFreshness(Enum):
    """Levels of data freshness requirements"""

    REALTIME = "realtime"  # Updated continuously
    NEAR_REALTIME = "near_realtime"  # Updated every few minutes
    HOURLY = "hourly"  # Updated hourly
    DAILY = "daily"  # Updated daily
    WEEKLY = "weekly"  # Updated weekly


@dataclass
class ExternalDataSource:
    """Configuration for an external data source"""

    id: str
    name: str
    source_type: DataSourceType
    endpoint: str
    api_key: Optional[str] = None
    refresh_interval: timedelta = timedelta(hours=1)
    freshness_requirement: DataFreshness = DataFreshness.DAILY
    last_updated: Optional[datetime] = None
    data: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IntegratedDataContext:
    """Context enriched with external data"""

    base_context: Dict[str, Any]
    external_data: Dict[DataSourceType, Dict[str, Any]]
    data_timestamps: Dict[DataSourceType, datetime]
    data_quality_scores: Dict[DataSourceType, float]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "base_context": self.base_context,
            "external_data": {k.value: v for k, v in self.external_data.items()},
            "data_timestamps": {k.value: v.isoformat() for k, v in self.data_timestamps.items()},
            "data_quality_scores": {k.value: v for k, v in self.data_quality_scores.items()},
        }


class WeatherData(BaseModel):
    """Weather data model"""

    location: str
    temperature: float
    feels_like: float
    humidity: int
    pressure: int
    wind_speed: float
    wind_direction: str
    condition: str
    condition_code: int
    visibility: float
    uv_index: float
    precipitation_probability: int
    sunrise: str
    sunset: str
    timestamp: datetime


class NewsArticle(BaseModel):
    """News article model"""

    title: str
    description: str
    url: str
    source: str
    published_at: datetime
    author: Optional[str] = None
    category: Optional[str] = None
    sentiment: Optional[float] = None  # -1 to 1 scale
    relevance_score: float = 0.0


class StockData(BaseModel):
    """Stock market data model"""

    symbol: str
    company_name: str
    current_price: float
    previous_close: float
    opening_price: float
    day_high: float
    day_low: float
    volume: int
    market_cap: Optional[float] = None
    pe_ratio: Optional[float] = None
    dividend_yield: Optional[float] = None
    timestamp: datetime


class TrafficData(BaseModel):
    """Traffic data model"""

    route_name: str
    current_travel_time: int  # minutes
    normal_travel_time: int  # minutes
    delay: int  # minutes
    congestion_level: str  # low, medium, heavy
    incidents: List[str]
    timestamp: datetime


class ExternalDataIntegrator:
    """
    Integrates external data sources to enhance JARVIS's contextual awareness.

    Features:
    - Multi-source data integration
    - Data quality assessment
    - Caching and freshness management
    - Error handling and fallbacks
    - Privacy-preserving data handling
    """

    def __init__(self, storage_path: str = "data/external_data"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.data_sources: Dict[str, ExternalDataSource] = {}
        self._cache = tool_cache
        self._http_client: Optional[httpx.AsyncClient] = None
        self._initialized = False
        self._lock = asyncio.Lock()

        # Initialize default data sources
        self._initialize_default_sources()

    def _initialize_default_sources(self):
        """Initialize default external data sources"""
        # Weather API (OpenWeatherMap)
        self.data_sources["openweathermap"] = ExternalDataSource(
            id="openweathermap",
            name="OpenWeatherMap",
            source_type=DataSourceType.WEATHER,
            endpoint="https://api.openweathermap.org/data/2.5/weather",
            refresh_interval=timedelta(minutes=30),
            freshness_requirement=DataFreshness.NEAR_REALTIME,
        )

        # News API
        self.data_sources["newsapi"] = ExternalDataSource(
            id="newsapi",
            name="NewsAPI",
            source_type=DataSourceType.NEWS,
            endpoint="https://newsapi.org/v2/top-headlines",
            refresh_interval=timedelta(hours=1),
            freshness_requirement=DataFreshness.HOURLY,
        )

        # Alpha Vantage for stock data
        self.data_sources["alphavantage"] = ExternalDataSource(
            id="alphavantage",
            name="Alpha Vantage",
            source_type=DataSourceType.STOCK_MARKET,
            endpoint="https://www.alphavantage.co/query",
            refresh_interval=timedelta(minutes=15),
            freshness_requirement=DataFreshness.NEAR_REALTIME,
        )

        # Generic traffic API
        self.data_sources["traffic"] = ExternalDataSource(
            id="traffic",
            name="Traffic Data",
            source_type=DataSourceType.TRAFFIC,
            endpoint="https://api.trafficdata.com/v1/routes",
            refresh_interval=timedelta(minutes=5),
            freshness_requirement=DataFreshness.REALTIME,
        )

        # Health data (placeholder)
        self.data_sources["health"] = ExternalDataSource(
            id="health",
            name="Health Data",
            source_type=DataSourceType.HEALTH,
            endpoint="https://api.healthdata.com/v1/user",
            refresh_interval=timedelta(hours=1),
            freshness_requirement=DataFreshness.HOURLY,
        )

    async def initialize(self):
        """Initialize the external data integrator"""
        if self._initialized:
            return

        # Initialize HTTP client
        self._http_client = httpx.AsyncClient(timeout=30.0)

        # Load saved data sources
        await self._load_data_sources()

        self._initialized = True
        log.info("External data integrator initialized")

    async def _load_data_sources(self):
        """Load data source configurations"""
        config_file = self.storage_path / "data_sources.json"

        if config_file.exists():
            try:
                with open(config_file, "r") as f:
                    data = json.load(f)

                for source_data in data.get("sources", []):
                    source = ExternalDataSource(
                        id=source_data["id"],
                        name=source_data["name"],
                        source_type=DataSourceType(source_data["source_type"]),
                        endpoint=source_data["endpoint"],
                        api_key=source_data.get("api_key"),
                        refresh_interval=timedelta(
                            seconds=source_data.get("refresh_interval", 3600)
                        ),
                        freshness_requirement=DataFreshness(
                            source_data.get("freshness_requirement", "daily")
                        ),
                        last_updated=datetime.fromisoformat(source_data["last_updated"])
                        if source_data.get("last_updated")
                        else None,
                        data=source_data.get("data", {}),
                        enabled=source_data.get("enabled", True),
                        metadata=source_data.get("metadata", {}),
                    )
                    self.data_sources[source.id] = source

            except Exception as e:
                log.error(f"Error loading data sources: {e}")

    async def _save_data_sources(self):
        """Save data source configurations"""
        config_file = self.storage_path / "data_sources.json"

        with open(config_file, "w") as f:
            json.dump(
                {
                    "sources": [
                        {
                            "id": source.id,
                            "name": source.name,
                            "source_type": source.source_type.value,
                            "endpoint": source.endpoint,
                            "api_key": source.api_key,
                            "refresh_interval": source.refresh_interval.total_seconds(),
                            "freshness_requirement": source.freshness_requirement.value,
                            "last_updated": source.last_updated.isoformat()
                            if source.last_updated
                            else None,
                            "data": source.data,
                            "enabled": source.enabled,
                            "metadata": source.metadata,
                        }
                        for source in self.data_sources.values()
                    ]
                },
                f,
                indent=2,
            )

    async def add_data_source(
        self,
        source_id: str,
        name: str,
        source_type: DataSourceType,
        endpoint: str,
        api_key: Optional[str] = None,
        refresh_interval: timedelta = timedelta(hours=1),
        freshness_requirement: DataFreshness = DataFreshness.DAILY,
    ) -> bool:
        """Add a new external data source"""
        async with self._lock:
            if source_id in self.data_sources:
                return False

            source = ExternalDataSource(
                id=source_id,
                name=name,
                source_type=source_type,
                endpoint=endpoint,
                api_key=api_key,
                refresh_interval=refresh_interval,
                freshness_requirement=freshness_requirement,
            )

            self.data_sources[source_id] = source
            await self._save_data_sources()
            return True

    async def remove_data_source(self, source_id: str) -> bool:
        """Remove an external data source"""
        async with self._lock:
            if source_id not in self.data_sources:
                return False

            del self.data_sources[source_id]
            await self._save_data_sources()
            return True

    async def get_integrated_context(
        self,
        base_context: Dict[str, Any],
        required_sources: Optional[Set[DataSourceType]] = None,
    ) -> IntegratedDataContext:
        """Get context enriched with external data"""
        if not self._initialized:
            await self.initialize()

        external_data = {}
        data_timestamps = {}
        data_quality_scores = {}

        # Determine which sources to fetch
        sources_to_fetch = []
        if required_sources:
            sources_to_fetch = [
                source
                for source in self.data_sources.values()
                if source.enabled and source.source_type in required_sources
            ]
        else:
            sources_to_fetch = [source for source in self.data_sources.values() if source.enabled]

        # Fetch data from each source
        for source in sources_to_fetch:
            try:
                data, quality_score = await self._fetch_source_data(source)
                if data:
                    external_data[source.source_type] = data
                    data_timestamps[source.source_type] = source.last_updated or datetime.now()
                    data_quality_scores[source.source_type] = quality_score
            except Exception as e:
                log.error(f"Error fetching data from {source.name}: {e}")
                # Use cached data if available
                if source.data:
                    external_data[source.source_type] = source.data
                    data_timestamps[source.source_type] = source.last_updated or datetime.now()
                    data_quality_scores[source.source_type] = (
                        0.5  # Lower quality score for cached data
                    )

        return IntegratedDataContext(
            base_context=base_context,
            external_data=external_data,
            data_timestamps=data_timestamps,
            data_quality_scores=data_quality_scores,
        )

    async def _fetch_source_data(
        self, source: ExternalDataSource
    ) -> Tuple[Optional[Dict[str, Any]], float]:
        """Fetch data from a specific source"""
        if not self._http_client:
            raise RuntimeError("HTTP client not initialized")

        # Check if data is still fresh
        if source.last_updated:
            time_since_update = datetime.now() - source.last_updated
            if time_since_update < source.refresh_interval:
                # Data is still fresh, return cached data
                return source.data, 0.9

        # Fetch new data
        try:
            data = None
            quality_score = 0.0

            if source.source_type == DataSourceType.WEATHER:
                data, quality_score = await self._fetch_weather_data(source)
            elif source.source_type == DataSourceType.NEWS:
                data, quality_score = await self._fetch_news_data(source)
            elif source.source_type == DataSourceType.STOCK_MARKET:
                data, quality_score = await self._fetch_stock_data(source)
            elif source.source_type == DataSourceType.TRAFFIC:
                data, quality_score = await self._fetch_traffic_data(source)
            elif source.source_type == DataSourceType.HEALTH:
                data, quality_score = await self._fetch_health_data(source)
            else:
                # Generic API fetch for other sources
                data, quality_score = await self._fetch_generic_data(source)

            if data:
                # Update source with new data
                source.data = data
                source.last_updated = datetime.now()
                await self._save_data_sources()

            return data, quality_score

        except Exception as e:
            log.error(f"Error fetching data from {source.name}: {e}")
            # Return cached data if available, with reduced quality score
            if source.data:
                return source.data, 0.5
            return None, 0.0

    async def _fetch_weather_data(
        self, source: ExternalDataSource
    ) -> Tuple[Optional[Dict[str, Any]], float]:
        """Fetch weather data from OpenWeatherMap or similar service"""
        if not source.api_key:
            # Try to get from environment
            import os

            source.api_key = os.getenv("OPENWEATHERMAP_API_KEY")

        if not source.api_key:
            log.warning("No API key for weather data")
            return None, 0.0

        try:
            # Get location from context or use default
            location = "London"  # Default
            if source.metadata.get("location"):
                location = source.metadata["location"]

            # Geocode location if needed
            lat, lon = await self._geocode_location(location)

            params = {"lat": lat, "lon": lon, "appid": source.api_key, "units": "metric"}

            response = await self._http_client.get(source.endpoint, params=params)
            response.raise_for_status()

            weather_data = response.json()

            # Parse and structure the data
            parsed_data = {
                "location": location,
                "latitude": lat,
                "longitude": lon,
                "temperature": weather_data["main"]["temp"],
                "feels_like": weather_data["main"]["feels_like"],
                "humidity": weather_data["main"]["humidity"],
                "pressure": weather_data["main"]["pressure"],
                "wind_speed": weather_data["wind"].get("speed", 0),
                "wind_direction": weather_data["wind"].get("deg", 0),
                "condition": weather_data["weather"][0]["main"],
                "condition_description": weather_data["weather"][0]["description"],
                "visibility": weather_data.get("visibility", 10000) / 1000,  # km
                "sunrise": datetime.fromtimestamp(weather_data["sys"]["sunrise"]).isoformat(),
                "sunset": datetime.fromtimestamp(weather_data["sys"]["sunset"]).isoformat(),
            }

            return parsed_data, 0.9

        except Exception as e:
            log.error(f"Error fetching weather data: {e}")
            return None, 0.0

    async def _geocode_location(self, location: str) -> Tuple[float, float]:
        """Geocode a location to latitude and longitude"""
        # This is a simplified implementation
        # In a real system, you would use a geocoding service
        default_coords = {
            "London": (51.5074, -0.1278),
            "New York": (40.7128, -74.0060),
            "Tokyo": (35.6762, 139.6503),
            "Paris": (48.8566, 2.3522),
            "Berlin": (52.5200, 13.4050),
        }

        coords = default_coords.get(location, (51.5074, -0.1278))  # Default to London
        return coords

    async def _fetch_news_data(
        self, source: ExternalDataSource
    ) -> Tuple[Optional[Dict[str, Any]], float]:
        """Fetch news data"""
        if not source.api_key:
            # Try to get from environment
            import os

            source.api_key = os.getenv("NEWSAPI_KEY")

        if not source.api_key:
            log.warning("No API key for news data")
            return None, 0.0

        try:
            params = {
                "apiKey": source.api_key,
                "country": source.metadata.get("country", "us"),
                "pageSize": 10,
            }

            # Add category if specified
            if source.metadata.get("category"):
                params["category"] = source.metadata["category"]

            response = await self._http_client.get(source.endpoint, params=params)
            response.raise_for_status()

            news_data = response.json()

            # Parse articles
            articles = []
            for article in news_data.get("articles", [])[:10]:  # Limit to 10 articles
                articles.append(
                    {
                        "title": article["title"],
                        "description": article["description"] or "",
                        "url": article["url"],
                        "source": article["source"]["name"],
                        "published_at": article["publishedAt"],
                        "author": article.get("author"),
                    }
                )

            return {"articles": articles, "total_results": news_data.get("totalResults", 0)}, 0.8

        except Exception as e:
            log.error(f"Error fetching news data: {e}")
            return None, 0.0

    async def _fetch_stock_data(
        self, source: ExternalDataSource
    ) -> Tuple[Optional[Dict[str, Any]], float]:
        """Fetch stock market data"""
        if not source.api_key:
            # Try to get from environment
            import os

            source.api_key = os.getenv("ALPHAVANTAGE_API_KEY")

        if not source.api_key:
            log.warning("No API key for stock data")
            return None, 0.0

        try:
            # Get symbols from metadata or use defaults
            symbols = source.metadata.get("symbols", ["AAPL", "GOOGL", "MSFT"])

            stock_data = {}
            for symbol in symbols[:5]:  # Limit to 5 symbols
                params = {
                    "function": "GLOBAL_QUOTE",
                    "symbol": symbol,
                    "apikey": source.api_key,
                }

                response = await self._http_client.get(source.endpoint, params=params)
                response.raise_for_status()

                quote_data = response.json()

                if "Global Quote" in quote_data:
                    quote = quote_data["Global Quote"]
                    stock_data[symbol] = {
                        "price": float(quote["05. price"]),
                        "change": float(quote["09. change"]),
                        "change_percent": quote["10. change percent"].rstrip("%"),
                        "volume": int(quote["06. volume"]),
                        "latest_trading_day": quote["07. latest trading day"],
                    }

            return stock_data, 0.85

        except Exception as e:
            log.error(f"Error fetching stock data: {e}")
            return None, 0.0

    async def _fetch_traffic_data(
        self, source: ExternalDataSource
    ) -> Tuple[Optional[Dict[str, Any]], float]:
        """Fetch traffic data"""
        try:
            # Get routes from metadata
            routes = source.metadata.get("routes", ["home_to_work"])

            traffic_data = {}
            for route in routes[:3]:  # Limit to 3 routes
                # This is a simulation - in reality you would call a traffic API
                # like Google Maps, HERE, or TomTom
                traffic_data[route] = {
                    "current_travel_time": 30 + hash(route) % 30,  # 30-60 minutes
                    "normal_travel_time": 25 + hash(route) % 20,  # 25-45 minutes
                    "delay": 5 + hash(route) % 25,  # 5-30 minutes
                    "congestion_level": ["low", "medium", "heavy"][hash(route) % 3],
                    "incidents": [] if hash(route) % 2 == 0 else ["accident on highway"],
                }

            return traffic_data, 0.7

        except Exception as e:
            log.error(f"Error fetching traffic data: {e}")
            return None, 0.0

    async def _fetch_health_data(
        self, source: ExternalDataSource
    ) -> Tuple[Optional[Dict[str, Any]], float]:
        """Fetch health data"""
        try:
            # This is a simulation - in reality you would connect to health APIs
            # like Apple Health, Google Fit, or wearable device APIs
            health_data = {
                "steps_today": 5000 + hash(datetime.now().day) % 10000,
                "heart_rate": 70 + hash(datetime.now().minute) % 20,
                "sleep_hours": 7 + (hash(datetime.now().day) % 3) - 1,
                "calories_burned": 2000 + hash(datetime.now().day) % 1000,
                "water_intake": 1.5 + (hash(datetime.now().hour) % 10) * 0.1,
            }

            return health_data, 0.75

        except Exception as e:
            log.error(f"Error fetching health data: {e}")
            return None, 0.0

    async def _fetch_generic_data(
        self, source: ExternalDataSource
    ) -> Tuple[Optional[Dict[str, Any]], float]:
        """Fetch data from a generic API endpoint"""
        try:
            headers = {}
            if source.api_key:
                headers["Authorization"] = f"Bearer {source.api_key}"

            params = source.metadata.get("params", {})

            response = await self._http_client.get(source.endpoint, headers=headers, params=params)
            response.raise_for_status()

            return response.json(), 0.8

        except Exception as e:
            log.error(f"Error fetching generic data from {source.name}: {e}")
            return None, 0.0

    async def get_relevant_news(
        self, topics: List[str], max_articles: int = 5
    ) -> List[NewsArticle]:
        """Get news articles relevant to specific topics"""
        # Use web search tool for topic-specific news
        search_tool = WebSearchTool()
        articles = []

        for topic in topics[:3]:  # Limit to 3 topics
            try:
                result = await search_tool.execute(
                    query=f"latest news about {topic}", num_results=max_articles
                )

                if result.success:
                    for item in result.data.get("results", [])[:max_articles]:
                        article = NewsArticle(
                            title=item.get("title", ""),
                            description=item.get("content", "")[:200],
                            url=item.get("url", ""),
                            source=item.get("url", "").split("//")[1].split("/")[0],
                            published_at=datetime.now(),  # Approximate
                            relevance_score=min(1.0, len(topic) / 20.0),  # Simple relevance
                        )
                        articles.append(article)

            except Exception as e:
                log.error(f"Error searching news for {topic}: {e}")

        # Sort by relevance and limit
        articles.sort(key=lambda x: x.relevance_score, reverse=True)
        return articles[:max_articles]

    async def get_weather_impact_assessment(self, location: str = "current") -> Dict[str, Any]:
        """Assess how weather might impact user activities"""
        try:
            # Get weather data
            weather_context = await self.get_integrated_context(
                {"location": location}, {DataSourceType.WEATHER}
            )

            weather_data = weather_context.external_data.get(DataSourceType.WEATHER, {})
            if not weather_data:
                return {"impact": "unknown", "recommendations": []}

            temperature = weather_data.get("temperature", 20)
            condition = weather_data.get("condition", "Clear").lower()
            wind_speed = weather_data.get("wind_speed", 0)
            precipitation_prob = weather_data.get("precipitation_probability", 0)

            impact_level = "low"
            recommendations = []

            # Temperature-based recommendations
            if temperature > 30:
                impact_level = "high"
                recommendations.append("Stay hydrated and seek shade during outdoor activities")
                recommendations.append("Consider indoor alternatives for exercise")
            elif temperature < 0:
                impact_level = "high"
                recommendations.append("Dress warmly for outdoor activities")
                recommendations.append("Be cautious of icy conditions")

            # Condition-based recommendations
            if "rain" in condition or precipitation_prob > 70:
                impact_level = "medium" if impact_level == "low" else impact_level
                recommendations.append("Carry an umbrella or raincoat")
                recommendations.append("Consider indoor transportation options")
            elif "snow" in condition:
                impact_level = "high"
                recommendations.append("Allow extra travel time due to road conditions")
                recommendations.append("Wear appropriate footwear")

            # Wind-based recommendations
            if wind_speed > 20:
                impact_level = "medium" if impact_level == "low" else impact_level
                recommendations.append("Secure loose outdoor items")
                recommendations.append("Be cautious during outdoor activities")

            return {
                "impact": impact_level,
                "temperature": temperature,
                "condition": condition,
                "wind_speed": wind_speed,
                "precipitation_probability": precipitation_prob,
                "recommendations": recommendations,
            }

        except Exception as e:
            log.error(f"Error assessing weather impact: {e}")
            return {"impact": "unknown", "recommendations": []}

    async def get_market_sentiment(self, symbols: List[str]) -> Dict[str, Any]:
        """Get market sentiment for specific stocks"""
        try:
            # Get stock data
            stock_context = await self.get_integrated_context(
                {"symbols": symbols}, {DataSourceType.STOCK_MARKET}
            )

            stock_data = stock_context.external_data.get(DataSourceType.STOCK_MARKET, {})
            if not stock_data:
                return {"overall_sentiment": "neutral", "details": {}}

            sentiments = {}
            positive_count = 0
            negative_count = 0

            for symbol, data in stock_data.items():
                change_percent = float(data.get("change_percent", 0))

                if change_percent > 2:
                    sentiment = "very_positive"
                    positive_count += 1
                elif change_percent > 0:
                    sentiment = "positive"
                    positive_count += 1
                elif change_percent < -2:
                    sentiment = "very_negative"
                    negative_count += 1
                elif change_percent < 0:
                    sentiment = "negative"
                    negative_count += 1
                else:
                    sentiment = "neutral"

                sentiments[symbol] = {
                    "sentiment": sentiment,
                    "change_percent": change_percent,
                    "price": data.get("price", 0),
                }

            # Overall sentiment
            if positive_count > negative_count:
                overall_sentiment = "positive" if positive_count > len(symbols) * 0.6 else "mixed"
            elif negative_count > positive_count:
                overall_sentiment = "negative" if negative_count > len(symbols) * 0.6 else "mixed"
            else:
                overall_sentiment = "neutral"

            return {
                "overall_sentiment": overall_sentiment,
                "positive_count": positive_count,
                "negative_count": negative_count,
                "neutral_count": len(symbols) - positive_count - negative_count,
                "details": sentiments,
            }

        except Exception as e:
            log.error(f"Error getting market sentiment: {e}")
            return {"overall_sentiment": "unknown", "details": {}}

    async def get_commuting_advice(self, route_name: str = "default_route") -> Dict[str, Any]:
        """Get commuting advice based on traffic and weather data"""
        try:
            # Get traffic and weather data
            context = await self.get_integrated_context(
                {"route": route_name}, {DataSourceType.TRAFFIC, DataSourceType.WEATHER}
            )

            traffic_data = context.external_data.get(DataSourceType.TRAFFIC, {}).get(route_name, {})
            weather_data = context.external_data.get(DataSourceType.WEATHER, {})

            if not traffic_data:
                return {"advice": "insufficient_data", "recommendations": []}

            current_time = traffic_data.get("current_travel_time", 30)
            normal_time = traffic_data.get("normal_travel_time", 25)
            delay = traffic_data.get("delay", 5)
            congestion = traffic_data.get("congestion_level", "medium")

            advice = "normal"
            recommendations = []

            # Traffic-based advice
            delay_ratio = delay / normal_time if normal_time > 0 else 0

            if delay_ratio > 0.5:  # More than 50% delay
                advice = "significant_delay"
                recommendations.append("Leave earlier to avoid traffic")
                recommendations.append("Consider alternative routes or transportation methods")
            elif delay_ratio > 0.2:  # 20-50% delay
                advice = "moderate_delay"
                recommendations.append("Plan for extra travel time")
                recommendations.append("Check for traffic incidents")

            # Congestion-based advice
            if congestion == "heavy":
                advice = "heavy_congestion"
                recommendations.append("Avoid this route if possible")
                recommendations.append("Consider leaving at a different time")

            # Weather impact on commuting
            if weather_data:
                condition = weather_data.get("condition", "").lower()
                precipitation_prob = weather_data.get("precipitation_probability", 0)

                if "rain" in condition or precipitation_prob > 70:
                    recommendations.append("Allow extra time due to wet road conditions")
                    recommendations.append("Ensure good visibility (wipers, lights)")
                elif "snow" in condition:
                    recommendations.append("Drive cautiously - roads may be slippery")
                    recommendations.append("Allow significant extra travel time")

            return {
                "advice": advice,
                "current_travel_time": current_time,
                "normal_travel_time": normal_time,
                "delay": delay,
                "delay_percentage": round(delay_ratio * 100, 1),
                "congestion_level": congestion,
                "recommendations": recommendations,
            }

        except Exception as e:
            log.error(f"Error getting commuting advice: {e}")
            return {"advice": "error", "recommendations": []}

    async def get_data_quality_report(self) -> Dict[str, Any]:
        """Get a report on the quality of integrated data sources"""
        report = {
            "total_sources": len(self.data_sources),
            "enabled_sources": sum(1 for s in self.data_sources.values() if s.enabled),
            "data_freshness": {},
            "quality_scores": {},
            "last_updates": {},
        }

        for source_id, source in self.data_sources.items():
            if source.enabled:
                # Freshness assessment
                if source.last_updated:
                    age = datetime.now() - source.last_updated
                    if age < source.refresh_interval:
                        freshness = "fresh"
                    elif age < source.refresh_interval * 2:
                        freshness = "acceptable"
                    else:
                        freshness = "stale"
                else:
                    freshness = "never_updated"

                report["data_freshness"][source_id] = freshness
                report["quality_scores"][source_id] = source.metadata.get("quality_score", 0.0)
                report["last_updates"][source_id] = (
                    source.last_updated.isoformat() if source.last_updated else None
                )

        return report

    async def cleanup_old_data(self, max_age_days: int = 30) -> int:
        """Clean up old cached data"""
        async with self._lock:
            cleaned_count = 0
            cutoff_date = datetime.now() - timedelta(days=max_age_days)

            for source in self.data_sources.values():
                if source.last_updated and source.last_updated < cutoff_date:
                    source.data = {}
                    source.last_updated = None
                    cleaned_count += 1

            if cleaned_count > 0:
                await self._save_data_sources()

            return cleaned_count


# Global instance
_external_data_integrator: Optional[ExternalDataIntegrator] = None


async def get_external_data_integrator() -> ExternalDataIntegrator:
    """Get the global external data integrator instance"""
    global _external_data_integrator
    if _external_data_integrator is None:
        _external_data_integrator = ExternalDataIntegrator()
        await _external_data_integrator.initialize()
    return _external_data_integrator


__all__ = [
    "ExternalDataIntegrator",
    "IntegratedDataContext",
    "DataSourceType",
    "DataFreshness",
    "ExternalDataSource",
    "WeatherData",
    "NewsArticle",
    "StockData",
    "TrafficData",
    "get_external_data_integrator",
]
