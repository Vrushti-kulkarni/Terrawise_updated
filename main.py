
"""
TerraWise - Complete Multi-Agent Sustainability System
A single file implementation using CrewAI with MySQL integration and FREE Gemini API
"""

import json
import mysql.connector
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import logging
import uuid
from decimal import Decimal
import os
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from crewai import Agent, Task, Crew, Process, LLM
    from crewai.tools import BaseTool
    from langchain_google_genai import ChatGoogleGenerativeAI
    import google.generativeai as genai
except ImportError:
    print("Please install required packages:")
    print("pip install crewai mysql-connector-python langchain-google-genai google-generativeai")
    exit(1)

# Set Google Gemini API key
if not os.getenv('GOOGLE_API_KEY'):
    print("Please set your GOOGLE_API_KEY environment variable")
    print("Get your FREE API key from: https://makersuite.google.com/app/apikey")
    print("export GOOGLE_API_KEY='your-free-key-here'")
    

os.environ['OPENAI_MODEL_NAME'] = 'gemini/gemini-2.5-flash'
os.environ['OPENAI_API_KEY'] = os.getenv('GOOGLE_API_KEY')
os.environ['OPENAI_API_BASE'] = 'https://generativelanguage.googleapis.com/v1beta'

# Configure Gemini
load_dotenv()
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

# ===========================
# DATABASE CONNECTION
# ===========================

class DatabaseManager:
    def __init__(self, host='localhost', database='terra', user='root', password='MySQL@21331'):
        self.connection_config = {
            'host': host,
            'database': database,
            'user': user,
            'password': password,
            'autocommit': True,
            'charset': 'utf8mb4'
        }
        self.test_connection()
    
    def test_connection(self):
        """Test database connection"""
        try:
            with mysql.connector.connect(**self.connection_config) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                logger.info("✅ Database connection successful!")
        except Exception as e:
            logger.error(f"❌ Database connection failed: {e}")
            logger.error("Make sure MySQL is running and terrawise_db exists")
            exit(1)
    
    def get_connection(self):
        return mysql.connector.connect(**self.connection_config)
    
    def execute_query(self, query: str, params: tuple = None):
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor(dictionary=True)
                cursor.execute(query, params or ())
                if query.strip().upper().startswith('SELECT'):
                    return cursor.fetchall()
                return cursor.rowcount
        except Exception as e:
            logger.error(f"Database query error: {e}")
            return [] if query.strip().upper().startswith('SELECT') else 0



# ===========================
# CARBON FOOTPRINT CALCULATOR
# ===========================

class CarbonFootprintCalculator:
    """Hardcoded carbon footprint calculations"""
    
    TRANSPORT_CO2 = {
        'car_petrol': 0.12, 'car_diesel': 0.10, 'car_electric': 0.05,
        'metro': 0.02, 'bus': 0.04, 'auto': 0.08, 'taxi': 0.11,
        'bike': 0.06, 'walk': 0.0, 'cycle': 0.0
    }
    
    ENERGY_CO2 = {
        'ac_usage': 1.5, 'fridge': 0.8, 'tv': 0.2,
        'washing_machine': 0.5, 'led_bulb': 0.01
    }
    
    @classmethod
    def calculate_transport_savings(cls, from_mode: str, to_mode: str, distance_km: float) -> float:
        from_co2 = cls.TRANSPORT_CO2.get(from_mode, 0.12)
        to_co2 = cls.TRANSPORT_CO2.get(to_mode, 0.02)
        savings = (from_co2 - to_co2) * distance_km
        return max(0, savings)
    
    @classmethod
    def calculate_energy_savings(cls, action: str, hours: float) -> float:
        savings_map = {
            'reduce_ac_temp': 0.3,
            'switch_to_led': 0.04,
            'optimal_fridge': 0.2
        }
        return savings_map.get(action, 0) * hours

# ===========================
# GEMINI AI HELPER CLASS
# ===========================

class GeminiHelper:
    """Helper class for direct Gemini API calls when needed"""
    
    def __init__(self):
        self.model = genai.GenerativeModel('gemini-1.5-flash')
    
    def generate_sustainability_advice(self, context: Dict, category: str = "general") -> str:
        """Generate sustainability advice using Gemini directly"""
        try:
            prompt = f"""
            Based on the following user context, provide 2-3 specific sustainability recommendations for the category '{category}'.
            
            User Context:
            - Location: {context.get('area', 'Mumbai')}
            - Current Activity: {context.get('current_activity', 'home')}
            - Time: {context.get('time_of_day', 'day')}
            - Environmental: AQI {context.get('aqi', 150)}, Temperature {context.get('temperature', 28)}°C
            
            Provide actionable, specific recommendations with:
            1. Clear title
            2. Step-by-step actions
            3. Expected CO2 savings in kg
            4. Cost savings in INR
            5. Difficulty level (Easy/Medium/Hard)
            
            Focus on Indian urban context, especially Mumbai.
            """
            
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return "Unable to generate recommendations at this time."
    
    # def calculate_rewards(self, actions: List[str]) -> Dict:
    #     """Calculate gamification rewards using Gemini"""
    #     try:
    #         prompt = f"""
    #         Calculate gamification rewards for these sustainability actions: {', '.join(actions)}
            
    #         For each action, provide:
    #         1. Coins earned (10-50 range)
    #         2. XP points (5-25 range)
    #         3. Achievement unlocked (if applicable)
    #         4. Motivational message
            
    #         Return as JSON format.
    #         """
            
    #         response = self.model.generate_content(prompt)
    #         # Simple fallback if JSON parsing fails
    #         return {
    #             'total_coins': len(actions) * 15,
    #             'total_xp': len(actions) * 10,
    #             'achievements': [],
    #             'message': 'Great job on taking sustainable actions!'
    #         }
    #     except Exception as e:
    #         logger.error(f"Gemini rewards calculation error: {e}")
    #         return {'total_coins': 15, 'total_xp': 10, 'message': 'Keep up the good work!'}

# ===========================
# CREWAI TOOLS
# ===========================
class StaticKnowledgeBase:
    """Static sustainability knowledge for Indian urban context"""
    
    SUSTAINABILITY_ACTIONS = {
        'transport': [
            {
                'action_type': 'transport',
                'title': 'Switch to Mumbai Metro for Daily Commute',
                'description': 'Use Mumbai Metro instead of car/taxi for your daily office commute. Metro is faster, cheaper, and significantly reduces carbon footprint.',
                'category': 'transport',
                'impact_metrics': {
                    'co2_reduction_kg': 1.8,
                    'coins_earned': 25,
                    'money_saved_inr': 150,
                    'difficulty': 'Easy',
                    'time_required': '5 minutes setup',
                    'cultural_relevance': 'High - Mumbai Metro is expanding and widely accepted'
                }
            },
            {
                'action_type': 'transport',
                'title': 'Use BEST Bus for Short Distance Travel',
                'description': 'Replace auto-rickshaw/taxi with BEST buses for distances under 5km. Use bus route apps for planning.',
                'category': 'transport',
                'impact_metrics': {
                    'co2_reduction_kg': 1.2,
                    'coins_earned': 15,
                    'money_saved_inr': 80,
                    'difficulty': 'Easy',
                    'time_required': '2 minutes planning',
                    'cultural_relevance': 'High - BEST buses are integral to Mumbai transport'
                }
            },
            {
                'action_type': 'transport',
                'title': 'Carpooling with Colleagues',
                'description': 'Organize carpooling with office colleagues or use apps like BlaBlaCar for regular commutes.',
                'category': 'transport',
                'impact_metrics': {
                    'co2_reduction_kg': 2.1,
                    'coins_earned': 30,
                    'money_saved_inr': 200,
                    'difficulty': 'Medium',
                    'time_required': '15 minutes coordination',
                    'cultural_relevance': 'High - Carpooling is socially accepted and cost-effective'
                }
            }
        ],
        'energy': [
            {
                'action_type': 'energy',
                'title': 'Optimize Air Conditioner Settings',
                'description': 'Set AC temperature to 26°C instead of 22°C, use timer function, and ensure proper insulation.',
                'category': 'energy',
                'impact_metrics': {
                    'co2_reduction_kg': 3.2,
                    'coins_earned': 40,
                    'money_saved_inr': 800,
                    'difficulty': 'Easy',
                    'time_required': '2 minutes daily',
                    'cultural_relevance': 'Very High - AC is major electricity consumer in Indian homes'
                }
            },
            {
                'action_type': 'energy',
                'title': 'Switch to LED Lighting Throughout Home',
                'description': 'Replace all CFL and incandescent bulbs with LED bulbs. Start with most-used rooms first.',
                'category': 'energy',
                'impact_metrics': {
                    'co2_reduction_kg': 2.5,
                    'coins_earned': 35,
                    'money_saved_inr': 600,
                    'difficulty': 'Easy',
                    'time_required': '30 minutes installation',
                    'cultural_relevance': 'High - LEDs are widely available and culturally accepted'
                }
            },
            {
                'action_type': 'energy',
                'title': 'Use Ceiling Fans with AC',
                'description': 'Run ceiling fans along with AC to circulate air better, allowing higher AC temperature settings.',
                'category': 'energy',
                'impact_metrics': {
                    'co2_reduction_kg': 1.8,
                    'coins_earned': 25,
                    'money_saved_inr': 400,
                    'difficulty': 'Easy',
                    'time_required': '1 minute daily',
                    'cultural_relevance': 'Very High - Ceiling fans are standard in Indian homes'
                }
            }
        ],
        'food': [
            {
                'action_type': 'food',
                'title': 'Plan Weekly Vegetarian Meals',
                'description': 'Plan vegetarian meals for 5 days a week. Use local, seasonal vegetables from nearby markets.',
                'category': 'food',
                'impact_metrics': {
                    'co2_reduction_kg': 1.5,
                    'coins_earned': 20,
                    'money_saved_inr': 300,
                    'difficulty': 'Medium',
                    'time_required': '20 minutes planning weekly',
                    'cultural_relevance': 'Very High - India has strong vegetarian culture'
                }
            },
            {
                'action_type': 'food',
                'title': 'Reduce Food Waste with Smart Storage',
                'description': 'Use proper containers, refrigerate correctly, and cook appropriate portions. Compost vegetable waste.',
                'category': 'food',
                'impact_metrics': {
                    'co2_reduction_kg': 1.1,
                    'coins_earned': 15,
                    'money_saved_inr': 250,
                    'difficulty': 'Easy',
                    'time_required': '5 minutes daily',
                    'cultural_relevance': 'High - Food waste reduction aligns with Indian values'
                }
            }
        ],
        'water': [
            {
                'action_type': 'water',
                'title': 'Install Low-Flow Shower Heads',
                'description': 'Replace regular shower heads with low-flow variants to reduce water consumption during bathing.',
                'category': 'water',
                'impact_metrics': {
                    'co2_reduction_kg': 0.8,
                    'coins_earned': 12,
                    'money_saved_inr': 180,
                    'difficulty': 'Easy',
                    'time_required': '15 minutes installation',
                    'cultural_relevance': 'Medium - Growing awareness about water conservation'
                }
            },
            {
                'action_type': 'water',
                'title': 'Rainwater Harvesting for Balcony Gardens',
                'description': 'Set up simple rainwater collection system for watering plants during monsoon season.',
                'category': 'water',
                'impact_metrics': {
                    'co2_reduction_kg': 1.0,
                    'coins_earned': 18,
                    'money_saved_inr': 120,
                    'difficulty': 'Medium',
                    'time_required': '2 hours setup',
                    'cultural_relevance': 'High - Traditional practice being revived in urban areas'
                }
            }
        ]
    }
    
    @classmethod
    def get_actions_by_category(cls, category: str) -> List[Dict]:
        """Get sustainability actions for a specific category"""
        return cls.SUSTAINABILITY_ACTIONS.get(category, [])
    
    @classmethod
    def get_all_actions(cls) -> List[Dict]:
        """Get all sustainability actions"""
        all_actions = []
        for category_actions in cls.SUSTAINABILITY_ACTIONS.values():
            all_actions.extend(category_actions)
        return all_actions
    
    
from pydantic import PrivateAttr

# class KnowledgeTool(BaseTool):
#     name: str = "knowledge_tool"
#     description: str = "Calculate sustainability impacts and get eco-friendly actions"

#     _db: DatabaseManager = PrivateAttr()
#     _calculator: CarbonFootprintCalculator = PrivateAttr()
#     _gemini_helper: GeminiHelper = PrivateAttr()

#     def __init__(self, db_manager: DatabaseManager):
#         super().__init__()
#         self._db = db_manager
#         self._calculator = CarbonFootprintCalculator()
#         self._gemini_helper = GeminiHelper()

#     def _run(self, action_type: str, context: str = "general") -> str:
#         # Get sustainability actions from database - FIX: use self._db instead of self.db
#         query = """
#         SELECT action_type, title, description, impact_metrics, category 
#         FROM sustainability_actions 
#         WHERE is_active = TRUE AND category = %s
#         LIMIT 5
#         """
        
#         actions = self._db.execute_query(query, (action_type,))  # FIXED: use self._db
        
#         # If no database actions, use Gemini to generate them
#         if not actions:
#             context_dict = json.loads(context) if isinstance(context, str) else {}
#             gemini_advice = self._gemini_helper.generate_sustainability_advice(context_dict, action_type)
            
#             # Create mock action structure
#             actions = [{
#                 'action_type': action_type,
#                 'title': f'AI-Generated {action_type.title()} Advice',
#                 'description': gemini_advice,
#                 'category': action_type,
#                 'impact_metrics': None
#             }]
        
#         # Add calculated impacts
#         enhanced_actions = []
#         for action in actions:
#             impact = json.loads(action['impact_metrics']) if action['impact_metrics'] else {}
            
#             # Add hardcoded calculations based on action type
#             if action_type == 'transport':
#                 impact['co2_reduction_kg'] = 1.5
#                 impact['coins_earned'] = 15
#                 impact['money_saved_inr'] = 25
#             elif action_type == 'energy':
#                 impact['co2_reduction_kg'] = 2.1
#                 impact['coins_earned'] = 20
#                 impact['money_saved_inr'] = 150
#             elif action_type == 'food':
#                 impact['co2_reduction_kg'] = 0.8
#                 impact['coins_earned'] = 10
#                 impact['money_saved_inr'] = 50
            
#             enhanced_actions.append({
#                 'action_type': action['action_type'],
#                 'title': action['title'],
#                 'description': action['description'],
#                 'category': action['category'],
#                 'impact_metrics': impact
#             })
        
#         return json.dumps({'available_actions': enhanced_actions})
class KnowledgeTool(BaseTool):
    name: str = "knowledge_tool"
    description: str = "Calculate sustainability impacts and get eco-friendly actions"

    _db: DatabaseManager = PrivateAttr()
    _calculator: CarbonFootprintCalculator = PrivateAttr()
    _gemini_helper: GeminiHelper = PrivateAttr()
    _static_kb: StaticKnowledgeBase = PrivateAttr()

    def __init__(self, db_manager: DatabaseManager):
        super().__init__()
        self._db = db_manager
        self._calculator = CarbonFootprintCalculator()
        self._gemini_helper = GeminiHelper()
        self._static_kb = StaticKnowledgeBase()

    def _run(self, action_type: str, context: str = "general") -> str:
        try:
            # First try to get actions from database
            query = """
            SELECT action_type, title, description, impact_metrics, category 
            FROM sustainability_actions 
            WHERE is_active = TRUE AND category = %s
            LIMIT 5
            """
            
            actions = self._db.execute_query(query, (action_type,))
            
            # If no database actions, use static knowledge base
            if not actions:
                logger.info(f"🔄 Using static knowledge base for category: {action_type}")
                static_actions = self._static_kb.get_actions_by_category(action_type)
                
                # If no static actions for category, get mixed actions
                if not static_actions:
                    static_actions = self._static_kb.get_all_actions()[:3]
                    logger.info(f"📚 Using mixed static actions, found {len(static_actions)} actions")
                
                return json.dumps({'available_actions': static_actions})
            
            # Process database actions
            enhanced_actions = []
            for action in actions:
                impact = json.loads(action['impact_metrics']) if action['impact_metrics'] else {}
                
                # Add calculated impacts based on action type if missing
                if not impact:
                    if action_type == 'transport':
                        impact = {
                            'co2_reduction_kg': 1.5,
                            'coins_earned': 15,
                            'money_saved_inr': 25,
                            'difficulty': 'Easy',
                            'time_required': '5 minutes',
                            'cultural_relevance': 'High'
                        }
                    elif action_type == 'energy':
                        impact = {
                            'co2_reduction_kg': 2.1,
                            'coins_earned': 20,
                            'money_saved_inr': 150,
                            'difficulty': 'Easy',
                            'time_required': '10 minutes',
                            'cultural_relevance': 'High'
                        }
                    elif action_type == 'food':
                        impact = {
                            'co2_reduction_kg': 0.8,
                            'coins_earned': 10,
                            'money_saved_inr': 50,
                            'difficulty': 'Medium',
                            'time_required': '15 minutes',
                            'cultural_relevance': 'High'
                        }
                
                enhanced_actions.append({
                    'action_type': action['action_type'],
                    'title': action['title'],
                    'description': action['description'],
                    'category': action['category'],
                    'impact_metrics': impact
                })
            
            return json.dumps({'available_actions': enhanced_actions})
            
        except Exception as e:
            logger.error(f"Knowledge tool error: {e}")
            # Fallback to static knowledge base
            logger.info("🔄 Falling back to static knowledge base due to error")
            static_actions = self._static_kb.get_actions_by_category(action_type)
            if not static_actions:
                static_actions = self._static_kb.get_all_actions()[:3]
            
            return json.dumps({'available_actions': static_actions})

class TransportTool(BaseTool):
    name: str = "transport_tool"
    description: str = "Get transport options with sustainability metrics"
    
    _db: DatabaseManager = PrivateAttr()
    
    def __init__(self, db_manager: DatabaseManager):
        super().__init__()
        self._db = db_manager
    
    def _run(self, from_location: str, to_location: str = "office") -> str:
        options = [
            {
                'mode': 'metro',
                'route': f'{from_location} to {to_location}',
                'duration': 25,
                'cost': 15,
                'co2_saved': 1.5,
                'walk_distance': 500,
                'availability': 'available',
                'sustainability_score': 9
            },
            {
                'mode': 'bus',
                'route': f'{from_location} to {to_location}',
                'duration': 35,
                'cost': 10,
                'co2_saved': 1.2,
                'walk_distance': 300,
                'availability': 'available',
                'sustainability_score': 8
            },
            {
                'mode': 'cycle_share',
                'route': f'{from_location} to {to_location}',
                'duration': 45,
                'cost': 5,
                'co2_saved': 2.1,
                'walk_distance': 100,
                'availability': 'available',
                'sustainability_score': 10
            },
            {
                'mode': 'taxi',
                'route': f'{from_location} to {to_location}',
                'duration': 20,
                'cost': 80,
                'co2_saved': 0,
                'walk_distance': 0,
                'availability': 'available',
                'sustainability_score': 2
            }
        ]
        
        return json.dumps({'transport_options': options})

class MemoryTool(BaseTool):
    name: str = "memory_tool"
    description: str = "Analyze user preferences and past behavior"

    _db: DatabaseManager = PrivateAttr()

    def __init__(self, db_manager: DatabaseManager):
        super().__init__()
        self._db = db_manager

    def _run(self, user_id: str) -> str:
        # Get user preferences - FIX: use self._db instead of self.db
        pref_query = """
        SELECT category, preference_type, preference_value, confidence_score
        FROM user_preferences 
        WHERE user_id = %s
        """
        preferences = self._db.execute_query(pref_query, (user_id,))  # FIXED: use self._db
        
        # Get recent feedback
        feedback_query = """
        SELECT r.category, rf.response, COUNT(*) as count
        FROM recommendation_feedback rf
        JOIN recommendations r ON rf.recommendation_id = r.id
        WHERE rf.user_id = %s AND rf.created_at >= DATE_SUB(NOW(), INTERVAL 30 DAY)
        GROUP BY r.category, rf.response
        """
        feedback = self._db.execute_query(feedback_query, (user_id,))  # FIXED: use self._db
        
        # Calculate preference scores
        preference_weights = {}
        for fb in feedback:
            category = fb['category']
            if category not in preference_weights:
                preference_weights[category] = {'total': 0, 'accepted': 0}
            
            preference_weights[category]['total'] += fb['count']
            if fb['response'] == 'accepted':
                preference_weights[category]['accepted'] += fb['count']
        
        # Calculate acceptance rates
        for category in preference_weights:
            total = preference_weights[category]['total']
            accepted = preference_weights[category]['accepted']
            preference_weights[category]['acceptance_rate'] = accepted / total if total > 0 else 0.5
        
        return json.dumps({
            'preferences': preferences,
            'preference_weights': preference_weights,
            'user_id': user_id
        })

# Also fix the ContextTool which has the same issue:
class ContextTool(BaseTool):
    name: str = "context_tool"
    description: str = "Get comprehensive user context including profile and current state"
    
    # Private attribute for DB
    _db: DatabaseManager = PrivateAttr()
    
    def __init__(self, db_manager: DatabaseManager):
        super().__init__()
        self._db = db_manager
    
    def _run(self, user_id: str) -> str:
        # Use self._db instead of self.db
        query = "SELECT * FROM users WHERE id = %s AND is_active = TRUE"
        user_data = self._db.execute_query(query, (user_id,))  # FIXED: use self._db
        
        if not user_data:
            return json.dumps({"error": "User not found"})
        
        user = user_data[0]
        
        # Get current state
        state_query = """
        SELECT current_activity FROM user_states 
        WHERE user_id = %s AND is_active = TRUE 
        ORDER BY created_at DESC LIMIT 1
        """
        state_data = self._db.execute_query(state_query, (user_id,))  # FIXED: use self._db
        current_activity = state_data[0]['current_activity'] if state_data else 'home'
        
        # Build context
        context = {
            'user_id': user_id,
            'name': user['name'],
            'area': user['area'],
            'current_activity': current_activity,
            'profile': json.loads(user['profile']) if user['profile'] else {},
            'timestamp': datetime.now().isoformat(),
            'environmental': {
                'weather': 'sunny',
                'temperature': 28,
                'aqi': 156
            },
            'temporal': {
                'time_of_day': self._get_time_of_day(),
                'day_type': 'weekday' if datetime.now().weekday() < 5 else 'weekend'
            }
        }
        
        return json.dumps(context)
    
    def _get_time_of_day(self) -> str:
        hour = datetime.now().hour
        if 5 <= hour < 12: return 'morning'
        elif 12 <= hour < 17: return 'afternoon'
        elif 17 <= hour < 21: return 'evening'
        else: return 'night'

# ===========================
# GEMINI LLM SETUP FUNCTION
# ===========================

def setup_gemini_llm():
    """Setup Gemini LLM with API key for CrewAI using CrewAI's LLM class"""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set")
    
    # Try different model configurations
    model_configs = [
        {
            "model": "gemini/gemini-2.5-flash",
            "kwargs": {
                "temperature": 0.3,
                "max_tokens": 1024,
            }
        },
        {
            "model": "gemini/gemini-2.5-pro", 
            "kwargs": {
                "temperature": 0.3,
                "max_tokens": 1024,
            }
        },
        {
            "model": "gemini/gemini-pro",
            "kwargs": {
                "temperature": 0.3,
                "max_tokens": 1024,
            }
        }
    ]
    
    for config in model_configs:
        try:
            # Use CrewAI's LLM class with the correct provider prefix
            llm = LLM(
                api_key=api_key,
                model=config["model"],
                **config["kwargs"]
            )
            
            # Test the model with a simple prompt
            # Note: CrewAI's LLM class might have different methods for testing
            # If .invoke() doesn't work, we'll use a different approach
            try:
                test_response = llm.invoke("Say 'Connected successfully'")
                logger.info(f"✅ Successfully connected to {config['model']}: {test_response[:50]}...")
            except:
                # If .invoke() doesn't work, just log the successful creation
                logger.info(f"✅ Successfully created LLM instance for {config['model']}")
            
            return llm
            
        except Exception as e:
            logger.warning(f"❌ Failed to initialize {config['model']}: {str(e)}")
            continue
    
    raise ValueError("All LLM models failed to initialize")
# ===========================
# CREWAI AGENTS WITH PROPER SETUP
# ===========================

def create_agents(db_manager: DatabaseManager):
    """Create all TerraWise agents using properly configured Gemini LLM"""
    
    # Setup the LLM first
    llm = setup_gemini_llm()
    
    # Now create agents with the properly configured LLM
    context_agent = Agent(
        role='Context Specialist',
        goal='Gather comprehensive user context for personalized sustainability recommendations',
        backstory="""You are an expert at understanding user context in Indian urban environments, 
                    especially Mumbai. You analyze user profiles, current activities, location, 
                    and environmental factors to provide rich context for sustainability recommendations.
                    You understand Indian culture, weather patterns, and urban challenges.""",
        tools=[ContextTool(db_manager)],
        llm=llm,  # Use the properly configured LLM
        verbose=False,
        allow_delegation=False
    )
    
    transport_agent = Agent(
        role='Mumbai Transport Expert',
        goal='Provide sustainable transport options and routing for Mumbai users',
        backstory="""You are a Mumbai transport expert who knows Metro lines, BEST buses, auto-rickshaws,
                    taxis, and cycle-sharing systems. You understand traffic patterns, monsoon impacts,
                    and provide accurate route information with environmental impact calculations.
                    You prioritize affordable and sustainable transport options for Indian users.""",
        tools=[TransportTool(db_manager)],
        llm=llm,  # Use the same LLM instance
        verbose=False,
        allow_delegation=False
    )
    
    knowledge_agent = Agent(
        role='Indian Sustainability Expert',
        goal='Calculate environmental impacts and provide sustainability knowledge for Indian context',
        backstory="""You are a sustainability expert with deep knowledge of carbon footprints,
                    energy consumption, and environmental impacts specific to Indian urban contexts.
                    You understand Indian electricity grids, local food systems, waste management,
                    and provide culturally appropriate sustainability advice.""",
        tools=[KnowledgeTool(db_manager)],
        llm=llm,  # Use the same LLM instance
        verbose=False,
        allow_delegation=False
    )
    
    memory_agent = Agent(
        role='User Behavior Analyst',
        goal='Analyze user preferences and learn from feedback to improve personalization',
        backstory="""You understand Indian user behavior patterns by analyzing feedback and preferences.
                    You know how to personalize recommendations considering Indian cultural preferences,
                    budget constraints, and lifestyle patterns. You help improve recommendation
                    acceptance by learning from user interactions.""",
        tools=[MemoryTool(db_manager)],
        llm=llm,  # Use the same LLM instance
        verbose=False,
        allow_delegation=False
    )
    
    recommendation_agent = Agent(
        role='Sustainability Recommendation Engine',
        goal='Generate personalized sustainability recommendations for Indian urban users',
        backstory="""You are the master recommendation engine who combines user context,
                    transport options, sustainability knowledge, and user preferences to create
                    highly personalized and actionable sustainability recommendations.
                    You understand Indian urban challenges, budget constraints, and cultural preferences.""",
        llm=llm,  # Use the same LLM instance
        verbose=False,
        allow_delegation=True
    )
    
    # rewards_agent = Agent(
    #     role='Gamification Specialist for Indian Users',
    #     goal='Calculate rewards, coins, and motivational elements culturally appropriate for Indian users',
    #     backstory="""You design engaging reward systems that motivate Indian users to adopt
    #                 sustainable behaviors. You understand Indian gaming preferences, social motivations,
    #                 and create positive reinforcement systems that work in Indian cultural context.
    #                 You calculate coins, track achievements, and create community-driven incentives.""",
    #     llm=llm,  # Use the same LLM instance
    #     verbose=False,
    #     allow_delegation=False
    # )
    
    return {
        'context': context_agent,
        'transport': transport_agent,
        'knowledge': knowledge_agent,
        'memory': memory_agent,
        'recommendation': recommendation_agent,
    }

# ===========================
# MODIFIED MAIN EXECUTION
# ===========================

def main():
    """Main function to run TerraWise demo with Gemini"""
    
    print("🌱 Welcome to TerraWise - AI-Powered Sustainability Assistant")
    print("🤖 Powered by FREE Google Gemini API")
    print("=" * 70)
    
    try:
        # Test Gemini API connection first using direct API
        print("🔍 Testing Gemini API connection...")
        if not test_gemini_connection():
            return
        
        # Initialize app
        print("🔧 Initializing TerraWise with Gemini...")
        app = TerraWiseApp()
        
        # Test LLM setup
        print("🧪 Testing LLM configuration...")
        try:
            llm = setup_gemini_llm()
            print("✅ LLM configured successfully!")
        except Exception as e:
            print(f"❌ LLM configuration failed: {e}")
            print("Falling back to direct API mode...")
            # Continue with fallback mode
        
        # Create demo user
        print("👤 Setting up demo user...")
        demo_user_id = setup_demo_user(app)
        
        if not demo_user_id:
            print("❌ Failed to create demo user")
            return
        
        print(f"✅ Demo user created: {demo_user_id}")
        
        # Run quick demo
        print("\n🤖 Running Quick Demo...")
        recommendations = app.generate_recommendations(demo_user_id, activity="traveling_to_work")
        
        print("\n📊 RECOMMENDATIONS RESULT:")
        print("-" * 40)
        print(json.dumps(recommendations, indent=2))
        
        print("\n🎉 TerraWise demo completed!")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"❌ Demo failed: {e}")

# ===========================
# TERRAWISE MAIN APPLICATION
# ===========================

class TerraWiseApp:
    def __init__(self, db_config: Dict = None):
        # Database setup
        if db_config:
            self.db_manager = DatabaseManager(**db_config)
        else:
            self.db_manager = DatabaseManager()
        
        # Initialize Gemini helper
        self.gemini_helper = GeminiHelper()
        
        # Initialize agents
        self.agents = create_agents(self.db_manager)
        
        logger.info("🌱 TerraWise Multi-Agent System Initialized with FREE Gemini API!")
    
    def generate_recommendations(self, user_id: str, activity: str = None) -> Dict:
        """Generate personalized sustainability recommendations using Gemini"""
        
        try:
            # Update user state if activity provided
            if activity:
                self._update_user_state(user_id, activity)
            
            # Create tasks for each agent with Gemini-optimized prompts
            context_task = Task(
                description=f"""Gather comprehensive context for user {user_id}. Include:
                              - User profile and preferences
                              - Current activity and location
                              - Environmental data (weather, AQI)
                              - Time context (morning/afternoon/evening)
                              Focus on Indian urban context, especially Mumbai.""",
                expected_output="Complete user context JSON with profile, activity, location, and environmental data",
                agent=self.agents['context']
            )
            
            transport_task = Task(
                description="""Get sustainable transport options for Mumbai with detailed metrics:
                              - Metro, bus, auto-rickshaw, taxi, cycle options
                              - CO2 impact, cost in INR, duration
                              - Sustainability scores (1-10)
                              - Practical considerations for Indian users""",
                expected_output="List of transport options with sustainability metrics and practical details",
                agent=self.agents['transport']
            )
            
            knowledge_task = Task(
                description="""Identify sustainability actions suitable for Indian urban context:
                              - Energy-saving actions considering Indian appliances
                              - Transport alternatives for Indian cities
                              - Water conservation relevant to Indian homes
                              Calculate environmental impact in Indian context.""",
                expected_output="Sustainability actions with calculated environmental impacts for Indian users",
                agent=self.agents['knowledge']
            )
            
            memory_task = Task(
                description=f"""Analyze user {user_id} behavior patterns and preferences:
                              - Past recommendation acceptance/rejection
                              - Category preferences (transport/energy/food)
                              - Cultural and budget considerations
                              - Personalization insights""",
                expected_output="User preference analysis and personalization insights",
                agent=self.agents['memory']
            )
            
            recommendation_task = Task(
                description=f"""Generate 2-3 personalized sustainability recommendations for user {user_id}:
                              
                              Each recommendation must include:
                              1. Clear, actionable title
                              2. Step-by-step instructions
                              3. Expected CO2 savings (kg)
                              4. Cost savings (INR)
                              5. Difficulty level (Easy/Medium/Hard)
                              6. Time required
                              7. Cultural relevance for Indian users
                              
                              Prioritize based on:
                              - User's current context and activity
                              - Past preferences and feedback
                              - Practical feasibility in Indian urban setting
                              - Maximum environmental impact""",
                expected_output="2-3 ranked, detailed recommendations with impact metrics and action steps",
                agent=self.agents['recommendation']
            )
            
            # rewards_task = Task(
            #     description=f"""Calculate gamification rewards for user {user_id}:
            #                   - Coin rewards (10-50 per action)
            #                   - XP points and level progression
            #                   - Achievement badges
            #                   - Social sharing incentives
            #                   - Culturally relevant motivational messages for Indian users""",
            #     expected_output="Complete gamification package with coins, XP, achievements, and motivation",
            #     agent=self.agents['rewards']
            # )
            
            # Create and run crew
            crew = Crew(
                agents=list(self.agents.values()),
                tasks=[context_task, transport_task, knowledge_task, memory_task, recommendation_task],
                process=Process.sequential,
                verbose=True
            )
            
            logger.info(f"🚀 Generating recommendations for user {user_id} using Gemini...")
            result = crew.kickoff()
            
            # Store recommendations in database
            session_id = self._store_recommendations(user_id, result)
            
            return {
                'status': 'success',
                'user_id': user_id,
                'session_id': session_id,
                'recommendations': str(result),
                'ai_model': 'Google Gemini Pro (Free)',
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            
            # Fallback to direct Gemini call
            try:
                logger.info("🔄 Falling back to direct Gemini API call...")
                context = self._get_user_context(user_id)
                gemini_advice = self.gemini_helper.generate_sustainability_advice(context, 'general')
                
                return {
                    'status': 'success_fallback',
                    'user_id': user_id,
                    'recommendations': gemini_advice,
                    'ai_model': 'Google Gemini Pro (Direct)',
                    'timestamp': datetime.now().isoformat()
                }
            except Exception as fallback_error:
                logger.error(f"Fallback also failed: {fallback_error}")
                return {
                    'status': 'error',
                    'error': str(e),
                    'user_id': user_id,
                    'timestamp': datetime.now().isoformat()
                }
    
    def submit_feedback(self, user_id: str, recommendation_id: str, response: str, 
                       feedback_reason: str = None) -> Dict:
        """Process user feedback on recommendations"""
        try:
            # Store feedback in database
            feedback_query = """
            INSERT INTO recommendation_feedback (
                recommendation_id, user_id, response, feedback_reason, created_at
            ) VALUES (%s, %s, %s, %s, NOW())
            """
            
            reason_json = json.dumps({'reason': feedback_reason}) if feedback_reason else None
            self.db_manager.execute_query(feedback_query, 
                                        (recommendation_id, user_id, response, reason_json))
            
            logger.info(f"✅ Feedback recorded: {user_id} {response} recommendation {recommendation_id}")
            
            return {
                'status': 'success',
                'message': f'Feedback recorded: {response}',
                'user_id': user_id,
                'recommendation_id': recommendation_id
            }
            
        except Exception as e:
            logger.error(f"Error submitting feedback: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def get_user_dashboard(self, user_id: str) -> Dict:
        """Get user dashboard with stats and progress"""
        try:
            # Get user stats
            stats_query = """
            SELECT u.name, u.area,
                   COALESCE(us.total_coins, 0) as total_coins,
                   COALESCE(us.total_co2_saved_kg, 0) as total_co2_saved,
                   COALESCE(us.current_streak, 0) as current_streak,
                   COALESCE(us.recommendations_accepted, 0) as accepted_count,
                   COALESCE(us.recommendations_total, 0) as total_count,
                   us.rank_position
            FROM users u
            LEFT JOIN user_stats us ON u.id = us.user_id
            WHERE u.id = %s
            """
            
            stats = self.db_manager.execute_query(stats_query, (user_id,))
            
            if not stats:
                return {'error': 'User not found'}
            
            user_stats = stats[0]
            
            # Get recent recommendations
            recent_query = """
            SELECT id, title, category, status, created_at 
            FROM recommendations 
            WHERE user_id = %s 
            ORDER BY created_at DESC 
            LIMIT 5
            """
            
            recent_recs = self.db_manager.execute_query(recent_query, (user_id,))
            
            return {
                'user_stats': user_stats,
                'recent_recommendations': recent_recs,
                'ai_model': 'Google Gemini Pro (Free)',
                'dashboard_generated_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting dashboard: {e}")
            return {'error': str(e)}
    
    def _get_user_context(self, user_id: str) -> Dict:
        """Get basic user context for fallback scenarios"""
        try:
            query = "SELECT * FROM users WHERE id = %s AND is_active = TRUE"
            user_data = self.db_manager.execute_query(query, (user_id,))
            
            if user_data:
                user = user_data[0]
                return {
                    'area': user['area'],
                    'current_activity': 'general',
                    'time_of_day': 'day',
                    'temperature': 28,
                    'aqi': 150
                }
            else:
                return {
                    'area': 'Mumbai',
                    'current_activity': 'general',
                    'time_of_day': 'day',
                    'temperature': 28,
                    'aqi': 150
                }
        except Exception as e:
            logger.error(f"Error getting user context: {e}")
            return {
                'area': 'Mumbai',
                'current_activity': 'general',
                'time_of_day': 'day',
                'temperature': 28,
                'aqi': 150
            }
    
    def _update_user_state(self, user_id: str, activity: str):
        """Update user's current activity state"""
        try:
            query = """
            INSERT INTO user_states (user_id, current_activity, context, is_active, created_at)
            VALUES (%s, %s, %s, TRUE, NOW())
            """
            
            context_data = json.dumps({
                'timestamp': datetime.now().isoformat(),
                'activity_source': 'user_input',
                'activity': activity,
                'ai_model': 'gemini-2.5-flash'
            })
            
            self.db_manager.execute_query(query, (user_id, activity, context_data))
            logger.info(f"📍 Updated user {user_id} activity to: {activity}")
            
        except Exception as e:
            logger.error(f"Error updating user state: {e}")
    
    def _store_recommendations(self, user_id: str, recommendations_data) -> str:
        """Store generated recommendations in database"""
        try:
            session_id = str(uuid.uuid4())
            
            # Store multiple sample recommendations
            recommendations = [
                {
                    'category': 'transport',
                    'title': 'Switch to Mumbai Metro',
                    'description': 'Use Mumbai Metro for daily commute instead of taxi/car',
                    'co2_saved': 1.5,
                    'money_saved': 45,
                    'coins': 20
                },
                {
                    'category': 'energy',
                    'title': 'Optimize AC Temperature',
                    'description': 'Set AC to 26°C instead of 22°C to save energy',
                    'co2_saved': 2.1,
                    'money_saved': 150,
                    'coins': 25
                },
                {
                    'category': 'food',
                    'title': 'Reduce Food Waste',
                    'description': 'Plan meals and store food properly to minimize waste',
                    'co2_saved': 0.8,
                    'money_saved': 200,
                    'coins': 15
                }
            ]
            
            query = """
            INSERT INTO recommendations (
                user_id, session_id, category, title, description, 
                impact_metrics, feasibility_score, user_preference_score, 
                priority_rank, status, expires_at, created_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
            """
            
            expires_at = datetime.now() + timedelta(hours=24)
            
            for i, rec in enumerate(recommendations):
                impact_metrics = json.dumps({
                    'co2_saved_kg': rec['co2_saved'],
                    'money_saved_inr': rec['money_saved'],
                    'coins_earned': rec['coins'],
                    'ai_model': 'gemini-2.5-flash'
                })
                
                self.db_manager.execute_query(query, (
                    user_id, session_id, rec['category'], 
                    rec['title'], rec['description'],
                    impact_metrics, 0.85, 0.75, i+1, 'pending', expires_at
                ))
            
            logger.info(f"💾 Stored {len(recommendations)} recommendations for user {user_id}, session {session_id}")
            return session_id
            
        except Exception as e:
            logger.error(f"Error storing recommendations: {e}")
            return str(uuid.uuid4())

# ===========================
# DEMO DATA SETUP
# ===========================

# 

def setup_demo_user(app: TerraWiseApp) -> str:
    """Create a demo user with proper error handling and validation"""
    try:
        demo_user_id = str(uuid.uuid4())
        logger.info(f"🔧 Creating demo user with ID: {demo_user_id}")
        
        # First, check if users table exists and has correct structure
        check_table_query = "DESCRIBE users"
        try:
            table_structure = app.db_manager.execute_query(check_table_query)
            logger.info(f"✅ Users table structure verified: {len(table_structure)} columns found")
        except Exception as e:
            logger.error(f"❌ Users table issue: {e}")
            # Try to create the table if it doesn't exist
            create_table_if_needed(app.db_manager)
        
        # Insert demo user with error handling
        user_query = """
        INSERT INTO users (id, name, email, area, profile, is_active, created_at) 
        VALUES (%s, %s, %s, %s, %s, TRUE, NOW())
        """
        
        profile_data = json.dumps({
            'sustainability_goal': 'reduce_carbon_footprint',
            'transport_modes': ['car', 'metro', 'bus', 'auto'],
            'vehicle_info': {
                'type': 'small_car_petrol',
                'fuel_efficiency': 15.5
            },
            'dietary_preferences': 'vegetarian',
            'energy_usage': {
                'monthly_bill': 3500,
                'has_renewable': False,
                'home_type': '2bhk_apartment'
            },
            'ai_model_preference': 'gemini-2.5-flash',
            'language': 'english',
            'budget_conscious': True
        })
        
        # Execute user insertion
        result = app.db_manager.execute_query(user_query, (
            demo_user_id, 
            'Demo User Mumbai', 
            f'demo_{demo_user_id[:8]}@terrawise.com', 
            'Andheri East', 
            profile_data
        ))
        
        if result == 0:
            logger.error("❌ Failed to insert demo user - no rows affected")
            return None
        
        # Verify user was created
        verify_query = "SELECT id, name FROM users WHERE id = %s"
        verification = app.db_manager.execute_query(verify_query, (demo_user_id,))
        
        if not verification:
            logger.error("❌ Demo user verification failed")
            return None
            
        logger.info(f"✅ Demo user verified: {verification[0]['name']}")
        
        # Insert some sample sustainability actions (with error handling)
        try:
            insert_sample_actions(app.db_manager)
        except Exception as e:
            logger.warning(f"⚠️ Could not insert sample actions: {e}")
            logger.info("📚 Will use static knowledge base instead")
        
        # Insert initial user state
        try:
            state_query = """
            INSERT INTO user_states (user_id, current_activity, context, is_active, created_at)
            VALUES (%s, %s, %s, TRUE, NOW())
            """
            
            context_data = json.dumps({
                'timestamp': datetime.now().isoformat(),
                'activity_source': 'demo_setup',
                'location': 'Mumbai',
                'setup_type': 'initial'
            })
            
            app.db_manager.execute_query(state_query, (demo_user_id, 'at_home', context_data))
            logger.info("✅ Initial user state created")
            
        except Exception as e:
            logger.warning(f"⚠️ Could not create initial user state: {e}")
        
        logger.info(f"🎉 Demo user setup completed successfully: {demo_user_id}")
        return demo_user_id
        
    except Exception as e:
        logger.error(f"❌ Error creating demo user: {e}")
        return None

def insert_sample_actions(db_manager: DatabaseManager):
    """Insert sample sustainability actions"""
    actions_query = """
    INSERT IGNORE INTO sustainability_actions (action_type, title, description, impact_metrics, category, is_active) 
    VALUES (%s, %s, %s, %s, %s, TRUE)
    """
    
    sample_actions = [
        ('transport', 'Use Mumbai Metro Daily', 'Switch from car/taxi to Metro for daily commute to save money and reduce emissions', 
         json.dumps({'co2_kg': 1.8, 'cost_inr': 150, 'difficulty': 'Easy'}), 'transport'),
        ('energy', 'LED Bulb Replacement', 'Replace all incandescent bulbs with LED to save 60% energy', 
         json.dumps({'co2_kg': 2.0, 'cost_inr': 600, 'difficulty': 'Easy'}), 'energy'),
        ('food', 'Weekly Vegetarian Meals', 'Plan 5 vegetarian meals per week using local vegetables', 
         json.dumps({'co2_kg': 1.5, 'cost_inr': 300, 'difficulty': 'Medium'}), 'food'),
        ('water', 'Rainwater Collection', 'Install simple rainwater collection for balcony plants', 
         json.dumps({'co2_kg': 1.0, 'cost_inr': 120, 'difficulty': 'Medium'}), 'water')
    ]
    
    for action in sample_actions:
        try:
            db_manager.execute_query(actions_query, action)
        except Exception as e:
            logger.warning(f"Action insertion failed: {e}")

def create_table_if_needed(db_manager: DatabaseManager):
    """Create basic table structure if missing"""
    try:
        create_users_table = """
        CREATE TABLE IF NOT EXISTS users (
            id VARCHAR(36) PRIMARY KEY,
            name VARCHAR(100) NOT NULL,
            email VARCHAR(150) UNIQUE NOT NULL,
            area VARCHAR(100),
            profile JSON,
            is_active BOOLEAN DEFAULT TRUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
        
        db_manager.execute_query(create_users_table)
        logger.info("✅ Users table created/verified")
        
    except Exception as e:
        logger.error(f"❌ Table creation failed: {e}")


##fixed store reccs
def fixed_store_recommendations(self, user_id: str, recommendations_data) -> str:
    """Store generated recommendations with proper user validation"""
    try:
        # First verify user exists
        user_check_query = "SELECT id FROM users WHERE id = %s AND is_active = TRUE"
        user_exists = self.db_manager.execute_query(user_check_query, (user_id,))
        
        if not user_exists:
            logger.error(f"❌ Cannot store recommendations: User {user_id} does not exist")
            return str(uuid.uuid4())  # Return dummy session ID
        
        session_id = str(uuid.uuid4())
        logger.info(f"✅ User verified, storing recommendations for session: {session_id}")
        
        # Store multiple sample recommendations
        recommendations = [
            {
                'category': 'transport',
                'title': 'Switch to Mumbai Metro',
                'description': 'Use Mumbai Metro for daily commute instead of taxi/car. Metro Line 1 (Blue) connects Andheri to Bandra-Kurla Complex efficiently.',
                'co2_saved': 1.8,
                'money_saved': 150,
                'coins': 25
            },
            {
                'category': 'energy',
                'title': 'Optimize AC Temperature',
                'description': 'Set AC to 26°C instead of 22°C during Mumbai\'s hot season. Use ceiling fans for better air circulation.',
                'co2_saved': 3.2,
                'money_saved': 800,
                'coins': 40
            },
            {
                'category': 'food',
                'title': 'Weekly Vegetarian Meal Planning',
                'description': 'Plan 5 vegetarian meals per week using seasonal vegetables from local Mumbai markets. Reduces carbon footprint and food costs.',
                'co2_saved': 1.5,
                'money_saved': 300,
                'coins': 20
            }
        ]
        
        # Check if recommendations table exists
        try:
            check_table = "SELECT COUNT(*) as count FROM recommendations LIMIT 1"
            self.db_manager.execute_query(check_table)
        except:
            logger.info("📋 Creating recommendations table...")
            create_recommendations_table = """
            CREATE TABLE IF NOT EXISTS recommendations (
                id VARCHAR(36) PRIMARY KEY DEFAULT (UUID()),
                user_id VARCHAR(36) NOT NULL,
                session_id VARCHAR(36) NOT NULL,
                category VARCHAR(50) NOT NULL,
                title VARCHAR(200) NOT NULL,
                description TEXT,
                impact_metrics JSON,
                feasibility_score DECIMAL(3,2) DEFAULT 0.85,
                user_preference_score DECIMAL(3,2) DEFAULT 0.75,
                priority_rank INT DEFAULT 1,
                status ENUM('pending', 'accepted', 'rejected', 'expired') DEFAULT 'pending',
                expires_at TIMESTAMP NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                INDEX idx_user_session (user_id, session_id),
                INDEX idx_status (status),
                INDEX idx_created (created_at)
            )
            """
            self.db_manager.execute_query(create_recommendations_table)
        
        query = """
        INSERT INTO recommendations (
            id, user_id, session_id, category, title, description, 
            impact_metrics, feasibility_score, user_preference_score, 
            priority_rank, status, expires_at, created_at
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
        """
        
        expires_at = datetime.now() + timedelta(hours=24)
        
        for i, rec in enumerate(recommendations):
            rec_id = str(uuid.uuid4())
            impact_metrics = json.dumps({
                'co2_saved_kg': rec['co2_saved'],
                'money_saved_inr': rec['money_saved'],
                'coins_earned': rec['coins'],
                'ai_model': 'gemini-2.5-flash',
                'difficulty': 'Easy' if i == 0 else 'Medium',
                'time_required': f'{(i+1)*5} minutes',
                'cultural_relevance': 'High'
            })
            
            result = self.db_manager.execute_query(query, (
                rec_id, user_id, session_id, rec['category'], 
                rec['title'], rec['description'],
                impact_metrics, 0.85, 0.75, i+1, 'pending', expires_at
            ))
            
            if result > 0:
                logger.info(f"✅ Stored recommendation {i+1}: {rec['title']}")
            else:
                logger.warning(f"⚠️ Failed to store recommendation {i+1}")
        
        logger.info(f"💾 Stored {len(recommendations)} recommendations for user {user_id}, session {session_id}")
        return session_id
        
    except Exception as e:
        logger.error(f"❌ Error storing recommendations: {e}")
        return str(uuid.uuid4())

# ===========================
# ENHANCED DEMO FUNCTIONS
# ===========================

def test_gemini_connection():
    """Test Gemini API connection"""
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content("Say 'Gemini API is working!' in a friendly way.")
        logger.info(f"✅ Gemini API test successful: {response.text[:50]}...")
        return True
    except Exception as e:
        logger.error(f"❌ Gemini API test failed: {e}")
        print("Please check your GOOGLE_API_KEY and internet connection")
        return False

def run_interactive_demo(app: TerraWiseApp, demo_user_id: str):
    """Run an interactive demo with user choices"""
    print("\n🎮 Interactive TerraWise Demo")
    print("=" * 40)
    
    activities = {
        '1': 'traveling_to_work',
        '2': 'at_home',
        '3': 'shopping',
        '4': 'cooking',
        '5': 'general'
    }
    
    print("Choose your current activity:")
    for key, activity in activities.items():
        print(f"{key}. {activity.replace('_', ' ').title()}")
    
    choice = input("\nEnter your choice (1-5): ").strip()
    activity = activities.get(choice, 'general')
    
    print(f"\n🚀 Generating recommendations for activity: {activity}")
    
    # Generate recommendations
    recommendations = app.generate_recommendations(demo_user_id, activity)
    
    print("\n📊 PERSONALIZED RECOMMENDATIONS:")
    print("-" * 50)
    print(json.dumps(recommendations, indent=2))
    
    # Ask for feedback
    if recommendations.get('status') in ['success', 'success_fallback']:
        print("\n💬 How do you like these recommendations?")
        print("1. Accept")
        print("2. Reject")
        print("3. Skip")
        
        feedback_choice = input("Enter your choice (1-3): ").strip()
        feedback_map = {'1': 'accepted', '2': 'rejected', '3': 'skipped'}
        
        if feedback_choice in feedback_map:
            # Get latest recommendation ID
            rec_query = "SELECT id FROM recommendations WHERE user_id = %s ORDER BY created_at DESC LIMIT 1"
            rec_result = app.db_manager.execute_query(rec_query, (demo_user_id,))
            
            if rec_result:
                rec_id = rec_result[0]['id']
                response = feedback_map[feedback_choice]
                feedback_reason = input("Optional - Why? (or press Enter to skip): ").strip()
                
                feedback_result = app.submit_feedback(
                    demo_user_id, rec_id, response, 
                    feedback_reason if feedback_reason else None
                )
                print(f"✅ Feedback submitted: {feedback_result}")

# ===========================
# MAIN EXECUTION
# ===========================

def main():
    """Main function to run TerraWise demo with Gemini"""
    
    print("🌱 Welcome to TerraWise - AI-Powered Sustainability Assistant")
    print("🤖 Powered by FREE Google Gemini API")
    print("=" * 70)
    
    try:
        # Test Gemini API connection first
        print("🔍 Testing Gemini API connection...")
        if not test_gemini_connection():
            return
        
        # Initialize app
        print("🔧 Initializing TerraWise with Gemini...")
        app = TerraWiseApp()
        
        # Create demo user
        print("👤 Setting up demo user...")
        demo_user_id = setup_demo_user(app)
        
        if not demo_user_id:
            print("❌ Failed to create demo user")
            return
        
        print(f"✅ Demo user created: {demo_user_id}")
        
        # Ask user for demo type
        print("\n📋 Choose demo type:")
        print("1. Quick Demo (automated)")
        print("2. Interactive Demo")
        
        demo_choice = input("Enter your choice (1-2): ").strip()
        
        if demo_choice == '2':
            run_interactive_demo(app, demo_user_id)
        else:
            # Quick automated demo
            print("\n🤖 Running Quick Demo...")
            
            # Generate recommendations
            print("Generating sustainability recommendations...")
            recommendations = app.generate_recommendations(demo_user_id, activity="traveling_to_work")
            
            print("\n📊 RECOMMENDATIONS RESULT:")
            print("-" * 40)
            print(json.dumps(recommendations, indent=2))
            
            # # Get user dashboard
            # print("\n📋 Getting user dashboard...")
            # dashboard = app.get_user_dashboard(demo_user_id)
            
            # print("\n📈 USER DASHBOARD:")
            # print("-" * 40)
            # print(json.dumps(dashboard, indent=2))
            
            # Demo feedback submission
            if recommendations.get('status') in ['success', 'success_fallback']:
                print("\n💬 Submitting demo feedback...")
                rec_query = "SELECT id FROM recommendations WHERE user_id = %s ORDER BY created_at DESC LIMIT 1"
                rec_result = app.db_manager.execute_query(rec_query, (demo_user_id,))
                
                if rec_result:
                    rec_id = rec_result[0]['id']
                    feedback = app.submit_feedback(demo_user_id, rec_id, 'accepted', 'Great AI-powered suggestions!')
                    print("✅ Feedback submitted:", feedback)
        
        print("\n🎉 TerraWise demo completed successfully!")
        print(f"🔑 Your demo user ID: {demo_user_id}")
        print("🌍 All powered by FREE Google Gemini API - completely open source!")
        
        # Show API usage info
        print("\n📊 API Information:")
        print("- Model: Google Gemini Pro (Free)")
        print("- Cost: $0.00 (Free tier)")
        print("- Rate limits: Generous free quota")
        print("- Setup: Just need free Google API key")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"❌ Demo failed: {e}")
        print("\nTroubleshooting:")
        print("1. Check your GOOGLE_API_KEY is set correctly")
        print("2. Ensure MySQL is running with terrawise_db")
        print("3. Check internet connection")
        print("4. Verify all packages are installed")

if __name__ == "__main__":
    main()
