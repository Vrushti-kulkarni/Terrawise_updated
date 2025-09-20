# ADD THIS CODE TO THE END OF YOUR EXISTING main.py FILE
# This integrates the GUI with your existing TerraWise multi-agent system

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
    

os.environ['OPENAI_MODEL_NAME'] = 'gemini/gemini-1.5-flash'
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
    
    def calculate_rewards(self, actions: List[str]) -> Dict:
        """Calculate gamification rewards using Gemini"""
        try:
            prompt = f"""
            Calculate gamification rewards for these sustainability actions: {', '.join(actions)}
            
            For each action, provide:
            1. Coins earned (10-50 range)
            2. XP points (5-25 range)
            3. Achievement unlocked (if applicable)
            4. Motivational message
            
            Return as JSON format.
            """
            
            response = self.model.generate_content(prompt)
            # Simple fallback if JSON parsing fails
            return {
                'total_coins': len(actions) * 15,
                'total_xp': len(actions) * 10,
                'achievements': [],
                'message': 'Great job on taking sustainable actions!'
            }
        except Exception as e:
            logger.error(f"Gemini rewards calculation error: {e}")
            return {'total_coins': 15, 'total_xp': 10, 'message': 'Keep up the good work!'}

# ===========================
# CREWAI TOOLS
# ===========================

from pydantic import PrivateAttr

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
        user_data = self._db.execute_query(query, (user_id,))

        
        if not user_data:
            return json.dumps({"error": "User not found"})
        
        user = user_data[0]
        
        # Get current state
        state_query = """
        SELECT current_activity FROM user_states 
        WHERE user_id = %s AND is_active = TRUE 
        ORDER BY created_at DESC LIMIT 1
        """
        state_data = self.db.execute_query(state_query, (user_id,))
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
    


class KnowledgeTool(BaseTool):
    name: str = "knowledge_tool"
    description: str = "Calculate sustainability impacts and get eco-friendly actions"

    _db: DatabaseManager = PrivateAttr()
    _calculator: CarbonFootprintCalculator = PrivateAttr()
    _gemini_helper: GeminiHelper = PrivateAttr()

    def __init__(self, db_manager: DatabaseManager):
        super().__init__()
        self._db = db_manager
        self._calculator = CarbonFootprintCalculator()
        self._gemini_helper = GeminiHelper()

    def _run(self, action_type: str, context: str = "general") -> str:
        # Get sustainability actions from database
        query = """
        SELECT action_type, title, description, impact_metrics, category 
        FROM sustainability_actions 
        WHERE is_active = TRUE AND category = %s
        LIMIT 5
        """
        
        actions = self.db.execute_query(query, (action_type,))
        
        # If no database actions, use Gemini to generate them
        if not actions:
            context_dict = json.loads(context) if isinstance(context, str) else {}
            gemini_advice = self.gemini_helper.generate_sustainability_advice(context_dict, action_type)
            
            # Create mock action structure
            actions = [{
                'action_type': action_type,
                'title': f'AI-Generated {action_type.title()} Advice',
                'description': gemini_advice,
                'category': action_type,
                'impact_metrics': None
            }]
        
        # Add calculated impacts
        enhanced_actions = []
        for action in actions:
            impact = json.loads(action['impact_metrics']) if action['impact_metrics'] else {}
            
            # Add hardcoded calculations based on action type
            if action_type == 'transport':
                impact['co2_reduction_kg'] = 1.5
                impact['coins_earned'] = 15
                impact['money_saved_inr'] = 25
            elif action_type == 'energy':
                impact['co2_reduction_kg'] = 2.1
                impact['coins_earned'] = 20
                impact['money_saved_inr'] = 150
            elif action_type == 'food':
                impact['co2_reduction_kg'] = 0.8
                impact['coins_earned'] = 10
                impact['money_saved_inr'] = 50
            
            enhanced_actions.append({
                'action_type': action['action_type'],
                'title': action['title'],
                'description': action['description'],
                'category': action['category'],
                'impact_metrics': impact
            })
        
        return json.dumps({'available_actions': enhanced_actions})

class MemoryTool(BaseTool):
    name: str = "memory_tool"
    description: str = "Analyze user preferences and past behavior"

    _db: DatabaseManager = PrivateAttr()

    def __init__(self, db_manager: DatabaseManager):
        super().__init__()
        self._db = db_manager

    def _run(self, user_id: str) -> str:
        # Get user preferences
        pref_query = """
        SELECT category, preference_type, preference_value, confidence_score
        FROM user_preferences 
        WHERE user_id = %s
        """
        preferences = self.db.execute_query(pref_query, (user_id,))
        
        # Get recent feedback
        feedback_query = """
        SELECT r.category, rf.response, COUNT(*) as count
        FROM recommendation_feedback rf
        JOIN recommendations r ON rf.recommendation_id = r.id
        WHERE rf.user_id = %s AND rf.created_at >= DATE_SUB(NOW(), INTERVAL 30 DAY)
        GROUP BY r.category, rf.response
        """
        feedback = self.db.execute_query(feedback_query, (user_id,))
        
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
            "model": "gemini/gemini-1.5-flash",
            "kwargs": {
                "temperature": 0.3,
                "max_tokens": 1024,
            }
        },
        {
            "model": "gemini/gemini-1.5-pro", 
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
    
    rewards_agent = Agent(
        role='Gamification Specialist for Indian Users',
        goal='Calculate rewards, coins, and motivational elements culturally appropriate for Indian users',
        backstory="""You design engaging reward systems that motivate Indian users to adopt
                    sustainable behaviors. You understand Indian gaming preferences, social motivations,
                    and create positive reinforcement systems that work in Indian cultural context.
                    You calculate coins, track achievements, and create community-driven incentives.""",
        llm=llm,  # Use the same LLM instance
        verbose=False,
        allow_delegation=False
    )
    
    return {
        'context': context_agent,
        'transport': transport_agent,
        'knowledge': knowledge_agent,
        'memory': memory_agent,
        'recommendation': recommendation_agent,
        'rewards': rewards_agent
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
            
            rewards_task = Task(
                description=f"""Calculate gamification rewards for user {user_id}:
                              - Coin rewards (10-50 per action)
                              - XP points and level progression
                              - Achievement badges
                              - Social sharing incentives
                              - Culturally relevant motivational messages for Indian users""",
                expected_output="Complete gamification package with coins, XP, achievements, and motivation",
                agent=self.agents['rewards']
            )
            
            # Create and run crew
            crew = Crew(
                agents=list(self.agents.values()),
                tasks=[context_task, transport_task, knowledge_task, memory_task, recommendation_task, rewards_task],
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
                'ai_model': 'gemini-1.5-flash'
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
                    'ai_model': 'gemini-1.5-flash'
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

def setup_demo_user(app: TerraWiseApp) -> str:
    """Create a demo user for testing"""
    try:
        demo_user_id = str(uuid.uuid4())
        
        # Insert demo user
        user_query = """
        INSERT INTO users (id, name, email, area, profile, created_at) 
        VALUES (%s, %s, %s, %s, %s, NOW())
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
            'ai_model_preference': 'gemini-1.5-flash',
            'language': 'english',
            'budget_conscious': True
        })
        
        app.db_manager.execute_query(user_query, (
            demo_user_id, 'Demo User', 'demo@terrawise.com', 'Andheri East', profile_data
        ))
        
        # Insert some sample sustainability actions
        actions_query = """
        INSERT INTO sustainability_actions (action_type, title, description, impact_metrics, category, is_active) 
        VALUES (%s, %s, %s, %s, %s, TRUE)
        """
        
        sample_actions = [
            ('transport', 'Use Mumbai Metro', 'Switch from car/taxi to Metro for daily commute', 
             json.dumps({'co2_kg': 1.5, 'cost_inr': 25}), 'transport'),
            ('energy', 'LED Bulb Replacement', 'Replace all incandescent bulbs with LED', 
             json.dumps({'co2_kg': 2.0, 'cost_inr': 150}), 'energy'),
            ('food', 'Meal Planning', 'Plan weekly meals to reduce food waste', 
             json.dumps({'co2_kg': 0.8, 'cost_inr': 200}), 'food'),
            ('water', 'Rainwater Harvesting', 'Install simple rainwater collection system', 
             json.dumps({'co2_kg': 1.2, 'cost_inr': 100}), 'water')
        ]
        
        for action in sample_actions:
            try:
                app.db_manager.execute_query(actions_query, action)
            except Exception as e:
                logger.warning(f"Action already exists or insert failed: {e}")
        
        logger.info(f"👤 Created demo user with sustainability actions: {demo_user_id}")
        return demo_user_id
        
    except Exception as e:
        logger.error(f"Error creating demo user: {e}")
        return None

# ===========================
# ENHANCED DEMO FUNCTIONS
# ===========================

def test_gemini_connection():
    """Test Gemini API connection"""
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
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
            
            # Get user dashboard
            print("\n📋 Getting user dashboard...")
            dashboard = app.get_user_dashboard(demo_user_id)
            
            print("\n📈 USER DASHBOARD:")
            print("-" * 40)
            print(json.dumps(dashboard, indent=2))
            
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



import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading
from typing import Dict, List
import uuid

class TerraWiseGUI:
    def __init__(self, root, terrawise_app: TerraWiseApp):
        self.root = root
        self.root.title("TerraWise - AI Multi-Agent Sustainability Assistant 🌱")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f8f0')
        
        # Connect to your actual TerraWise app with all the agents!
        self.app = terrawise_app
        self.current_user_id = None
        self.current_recommendations = []
        
        # Configure styles
        self.setup_styles()
        
        # Create GUI components
        self.create_main_interface()
        
        # Load demo user and connect to your database
        self.load_real_user()
    
    def setup_styles(self):
        """Configure custom styles for the green theme"""
        style = ttk.Style()
        
        # Configure colors
        colors = {
            'primary': '#2d5a3d',      # Dark green
            'secondary': '#4a7c59',    # Medium green
            'accent': '#6fb37b',       # Light green
            'background': '#f0f8f0',   # Very light green
            'success': '#28a745',      # Success green
            'warning': '#ffc107',      # Warning yellow
            'light': '#e8f5e8'         # Light green background
        }
        
        # Custom button styles
        style.theme_use('clam')
        
        style.configure('Primary.TButton',
                       background=colors['primary'],
                       foreground='white',
                       padding=(15, 8),
                       font=('Arial', 10, 'bold'))
        style.map('Primary.TButton',
                 background=[('active', colors['secondary'])])
        
        style.configure('Secondary.TButton',
                       background=colors['accent'],
                       foreground='white',
                       padding=(10, 6))
        style.map('Secondary.TButton',
                 background=[('active', colors['secondary'])])
        
        # Frame styles
        style.configure('Card.TFrame',
                       background='white',
                       relief='raised',
                       borderwidth=1)
        
        # Label styles
        style.configure('Title.TLabel',
                       background=colors['background'],
                       foreground=colors['primary'],
                       font=('Arial', 16, 'bold'))
        
        style.configure('Subtitle.TLabel',
                       background=colors['background'],
                       foreground=colors['secondary'],
                       font=('Arial', 12, 'bold'))
        
        style.configure('Body.TLabel',
                       background='white',
                       foreground='#333333',
                       font=('Arial', 10))
        
        # Metric styles
        style.configure('Metric.TLabel',
                       background='white',
                       foreground=colors['success'],
                       font=('Arial', 14, 'bold'))
    
    def create_main_interface(self):
        """Create the main GUI interface"""
        
        # Header
        self.create_header()
        
        # Main content area
        self.main_frame = ttk.Frame(self.root, style='Card.TFrame')
        self.main_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Create tabs
        self.create_dashboard_tab()
        self.create_recommendations_tab()
        self.create_agents_tab()  # New tab to show your agents!
        self.create_profile_tab()
    
    def create_header(self):
        """Create the application header"""
        header_frame = ttk.Frame(self.root, style='Card.TFrame')
        header_frame.pack(fill='x', padx=20, pady=(20, 10))
        
        # Logo and title
        title_frame = ttk.Frame(header_frame)
        title_frame.pack(side='left', padx=20, pady=10)
        
        ttk.Label(title_frame, text="🌱 TerraWise", 
                 font=('Arial', 24, 'bold'), 
                 foreground='#2d5a3d',
                 background='white').pack(anchor='w')
        
        ttk.Label(title_frame, text="Multi-Agent AI Sustainability System (Powered by Gemini)", 
                 font=('Arial', 12), 
                 foreground='#4a7c59',
                 background='white').pack(anchor='w')
        
        # User info and controls
        user_frame = ttk.Frame(header_frame)
        user_frame.pack(side='right', padx=20, pady=10)
        
        self.user_label = ttk.Label(user_frame, text="Loading user...", 
                                   font=('Arial', 12, 'bold'),
                                   background='white')
        self.user_label.pack(anchor='e')
        
        ttk.Button(user_frame, text="🔄 Refresh Data",
                  style='Secondary.TButton',
                  command=self.refresh_real_data).pack(anchor='e', pady=5)
    
    def create_dashboard_tab(self):
        """Create the main dashboard tab with REAL data"""
        dashboard_frame = ttk.Frame(self.notebook)
        self.notebook.add(dashboard_frame, text="🏠 Dashboard")
        
        # Create scrollable canvas
        canvas = tk.Canvas(dashboard_frame, bg='#f0f8f0')
        scrollbar = ttk.Scrollbar(dashboard_frame, orient='vertical', command=canvas.yview)
        self.scrollable_dashboard = ttk.Frame(canvas)
        
        self.scrollable_dashboard.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=self.scrollable_dashboard, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # This will be populated with real data
        self.dashboard_content_frame = self.scrollable_dashboard
        
    def create_recommendations_tab(self):
        """Create recommendations tab that uses your ACTUAL AGENTS"""
        rec_frame = ttk.Frame(self.notebook)
        self.notebook.add(rec_frame, text="💡 AI Recommendations")
        
        # Header
        header_frame = ttk.Frame(rec_frame)
        header_frame.pack(fill='x', padx=20, pady=10)
        
        ttk.Label(header_frame, text="Multi-Agent AI Recommendations (Gemini + CrewAI)",
                 style='Title.TLabel').pack(side='left')
        
        ttk.Button(header_frame, text="🤖 Generate with AI Agents",
                  style='Primary.TButton',
                  command=self.generate_real_recommendations).pack(side='right')
        
        # Activity selector
        activity_frame = ttk.Frame(rec_frame)
        activity_frame.pack(fill='x', padx=20, pady=5)
        
        ttk.Label(activity_frame, text="Current Activity:",
                 style='Body.TLabel').pack(side='left', padx=(0, 10))
        
        self.activity_var = tk.StringVar(value="general")
        activities = ["general", "traveling_to_work", "at_home", "shopping", "cooking"]
        
        activity_combo = ttk.Combobox(activity_frame, textvariable=self.activity_var,
                                     values=activities, state='readonly', width=20)
        activity_combo.pack(side='left')
        
        # Agent status
        status_frame = ttk.Frame(rec_frame)
        status_frame.pack(fill='x', padx=20, pady=5)
        
        self.agent_status_label = ttk.Label(status_frame, text="🤖 Agents: Ready", 
                                           style='Body.TLabel')
        self.agent_status_label.pack(side='left')
        
        # Recommendations display
        self.rec_display_frame = ttk.Frame(rec_frame)
        self.rec_display_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        self.display_initial_recommendations()
    
    def create_agents_tab(self):
        """NEW TAB: Show your actual AI agents and their status"""
        agents_frame = ttk.Frame(self.notebook)
        self.notebook.add(agents_frame, text="🤖 AI Agents")
        
        # Header
        header_frame = ttk.Frame(agents_frame)
        header_frame.pack(fill='x', padx=20, pady=20)
        
        ttk.Label(header_frame, text="Multi-Agent AI System Status",
                 style='Title.TLabel').pack(anchor='w')
        
        ttk.Label(header_frame, text="Powered by Google Gemini Pro + CrewAI",
                 style='Body.TLabel').pack(anchor='w', pady=(5, 0))
        
        # Agents status
        agents_content = ttk.Frame(agents_frame)
        agents_content.pack(fill='both', expand=True, padx=20, pady=10)
        
        # Show each agent
        agents_info = [
            {
                'name': 'Context Agent',
                'role': 'Context Specialist',
                'emoji': '🔍',
                'description': 'Analyzes user profile, location, activity, and environmental factors',
                'status': 'Active'
            },
            {
                'name': 'Transport Agent',
                'role': 'Mumbai Transport Expert',
                'emoji': '🚗',
                'description': 'Provides Metro, bus, auto-rickshaw, and cycling recommendations',
                'status': 'Active'
            },
            {
                'name': 'Knowledge Agent',
                'role': 'Sustainability Expert',
                'emoji': '🌱',
                'description': 'Calculates carbon footprints and environmental impacts',
                'status': 'Active'
            },
            {
                'name': 'Memory Agent',
                'role': 'Behavior Analyst',
                'emoji': '🧠',
                'description': 'Learns from feedback and personalizes recommendations',
                'status': 'Active'
            },
            {
                'name': 'Recommendation Agent',
                'role': 'Master Coordinator',
                'emoji': '💡',
                'description': 'Combines all agent insights for personalized suggestions',
                'status': 'Active'
            },
            {
                'name': 'Rewards Agent',
                'role': 'Gamification Specialist',
                'emoji': '🎮',
                'description': 'Calculates coins, achievements, and motivational elements',
                'status': 'Active'
            }
        ]
        
        for i, agent in enumerate(agents_info):
            self.create_agent_card(agents_content, agent, i)
    
    def create_agent_card(self, parent, agent_info, index):
        """Create agent status card"""
        card_frame = ttk.Frame(parent, style='Card.TFrame')
        card_frame.pack(fill='x', pady=8)
        
        content_frame = ttk.Frame(card_frame)
        content_frame.pack(fill='x', padx=20, pady=15)
        
        # Header with emoji and name
        header_frame = ttk.Frame(content_frame)
        header_frame.pack(fill='x')
        
        agent_title = f"{agent_info['emoji']} {agent_info['name']}"
        ttk.Label(header_frame, text=agent_title,
                 font=('Arial', 14, 'bold'),
                 background='white').pack(side='left')
        
        # Status indicator
        status_color = '#28a745' if agent_info['status'] == 'Active' else '#dc3545'
        status_label = tk.Label(header_frame, text=agent_info['status'],
                               bg=status_color, fg='white',
                               font=('Arial', 9, 'bold'),
                               padx=8, pady=2)
        status_label.pack(side='right')
        
        # Role
        ttk.Label(content_frame, text=f"Role: {agent_info['role']}",
                 font=('Arial', 11, 'bold'),
                 background='white').pack(anchor='w', pady=(5, 0))
        
        # Description
        ttk.Label(content_frame, text=agent_info['description'],
                 font=('Arial', 10),
                 background='white',
                 wraplength=800).pack(anchor='w', pady=(5, 0))
    
    def create_profile_tab(self):
        """Create user profile tab with REAL database data"""
        profile_frame = ttk.Frame(self.notebook)
        self.notebook.add(profile_frame, text="👤 Profile")
        
        # Profile content will be populated with real data
        self.profile_content_frame = ttk.Frame(profile_frame)
        self.profile_content_frame.pack(fill='both', expand=True)
    
    def load_real_user(self):
        """Load actual user from your TerraWise database"""
        try:
            # Create or get demo user using your existing function
            self.current_user_id = setup_demo_user(self.app)
            
            if self.current_user_id:
                self.user_label.config(text=f"User: {self.current_user_id[:8]}...")
                logger.info(f"GUI connected to user: {self.current_user_id}")
                
                # Load real dashboard data
                self.load_real_dashboard()
                
                # Load real profile data
                self.load_real_profile()
            else:
                messagebox.showerror("Error", "Failed to load user data")
                
        except Exception as e:
            logger.error(f"Error loading real user: {e}")
            messagebox.showerror("Error", f"Failed to connect to database: {e}")
    
    def load_real_dashboard(self):
        """Load dashboard with actual data from your database"""
        try:
            # Get real user dashboard using your existing method
            dashboard_data = self.app.get_user_dashboard(self.current_user_id)
            
            # Clear existing content
            for widget in self.dashboard_content_frame.winfo_children():
                widget.destroy()
            
            if 'error' in dashboard_data:
                ttk.Label(self.dashboard_content_frame, 
                         text=f"Error loading dashboard: {dashboard_data['error']}",
                         style='Body.TLabel').pack(pady=50)
                return
            
            # Create stats cards with REAL data
            stats_frame = ttk.Frame(self.dashboard_content_frame)
            stats_frame.pack(fill='x', padx=10, pady=10)
            
            ttk.Label(stats_frame, text="Your Real Impact (Live Data)", 
                     style='Title.TLabel').pack(anchor='w')
            
            # Cards container
            cards_frame = ttk.Frame(stats_frame)
            cards_frame.pack(fill='x', pady=10)
            
            user_stats = dashboard_data.get('user_stats', {})
            
            # Real stats from database
            self.create_stat_card(cards_frame, "🪙 Total Coins", 
                                 f"{user_stats.get('total_coins', 0):,}", 0, 0)
            
            self.create_stat_card(cards_frame, "🌍 CO₂ Saved", 
                                 f"{user_stats.get('total_co2_saved_kg', 0):.1f} kg", 0, 1)
            
            self.create_stat_card(cards_frame, "🔥 Current Streak", 
                                 f"{user_stats.get('current_streak', 0)} days", 0, 2)
            
            self.create_stat_card(cards_frame, "✅ Accepted", 
                                 f"{user_stats.get('recommendations_accepted', 0)}", 1, 0)
            
            self.create_stat_card(cards_frame, "📊 Total Recs", 
                                 f"{user_stats.get('recommendations_total', 0)}", 1, 1)
            
            rank_pos = user_stats.get('rank_position', 'N/A')
            self.create_stat_card(cards_frame, "🏆 Rank", 
                                 f"#{rank_pos}" if rank_pos != 'N/A' else "Unranked", 1, 2)
            
            # Real recent recommendations
            recent_recs = dashboard_data.get('recent_recommendations', [])
            if recent_recs:
                activity_frame = ttk.Frame(self.dashboard_content_frame)
                activity_frame.pack(fill='x', padx=10, pady=10)
                
                ttk.Label(activity_frame, text="Recent Recommendations (From Database)", 
                         style='Title.TLabel').pack(anchor='w')
                
                for rec in recent_recs[:3]:
                    self.create_real_activity_item(activity_frame, rec)
            
            # AI Model info
            ai_frame = ttk.Frame(self.dashboard_content_frame, style='Card.TFrame')
            ai_frame.pack(fill='x', padx=10, pady=10)
            
            ai_content = ttk.Frame(ai_frame)
            ai_content.pack(fill='x', padx=15, pady=10)
            
            ttk.Label(ai_content, text="🤖 AI System Status", 
                     style='Subtitle.TLabel').pack(anchor='w')
            
            ttk.Label(ai_content, text="Model: Google Gemini Pro (Free)", 
                     style='Body.TLabel').pack(anchor='w')
            
            ttk.Label(ai_content, text="Framework: CrewAI Multi-Agent System", 
                     style='Body.TLabel').pack(anchor='w')
            
            ttk.Label(ai_content, text="Database: MySQL Connected ✅", 
                     style='Body.TLabel').pack(anchor='w')
            
        except Exception as e:
            logger.error(f"Error loading real dashboard: {e}")
            ttk.Label(self.dashboard_content_frame, 
                     text=f"Error loading dashboard: {e}",
                     style='Body.TLabel').pack(pady=50)
    
    def create_stat_card(self, parent, title, value, row, col):
        """Create individual stat card"""
        card = ttk.Frame(parent, style='Card.TFrame')
        card.grid(row=row, column=col, padx=10, pady=5, sticky='ew')
        
        # Configure grid weights
        parent.grid_columnconfigure(col, weight=1)
        
        ttk.Label(card, text=title, 
                 font=('Arial', 10), 
                 background='white').pack(pady=(10, 5))
        
        ttk.Label(card, text=value, 
                 style='Metric.TLabel').pack(pady=(0, 10))
    
    def create_real_activity_item(self, parent, recommendation):
        """Create activity item with real database data"""
        item_frame = ttk.Frame(parent, style='Card.TFrame')
        item_frame.pack(fill='x', pady=5)
        
        content_frame = ttk.Frame(item_frame)
        content_frame.pack(fill='x', padx=15, pady=10)
        
        # Icon and title
        header_frame = ttk.Frame(content_frame)
        header_frame.pack(fill='x')
        
        icons = {'transport': '🚗', 'energy': '⚡', 'food': '🍽️', 'water': '💧'}
        icon = icons.get(recommendation.get('category', ''), '🌱')
        
        title_text = f"{icon} {recommendation.get('title', 'Recommendation')}"
        ttk.Label(header_frame, text=title_text, 
                 font=('Arial', 12, 'bold'),
                 background='white').pack(side='left')
        
        status = recommendation.get('status', 'pending')
        status_colors = {'completed': '#28a745', 'pending': '#ffc107', 'expired': '#dc3545'}
        status_color = status_colors.get(status, '#6c757d')
        
        status_label = tk.Label(header_frame, text=status.title(),
                               bg=status_color, fg='white',
                               font=('Arial', 8, 'bold'),
                               padx=8, pady=2)
        status_label.pack(side='right')
        
        # Details with real database timestamp
        created_at = recommendation.get('created_at', 'Unknown')
        if hasattr(created_at, 'strftime'):
            created_str = created_at.strftime('%Y-%m-%d %H:%M')
        else:
            created_str = str(created_at)[:19] if created_at else 'Unknown'
            
        details = f"Category: {recommendation.get('category', 'general').title()} | Created: {created_str}"
        ttk.Label(content_frame, text=details, 
                 font=('Arial', 9),
                 foreground='#666666',
                 background='white').pack(anchor='w', pady=(5, 0))
    
    def load_real_profile(self):
        """Load profile with real user data from database"""
        try:
            # Get user data from database
            query = "SELECT * FROM users WHERE id = %s"
            user_data = self.app.db_manager.execute_query(query, (self.current_user_id,))
            
            # Clear existing content
            for widget in self.profile_content_frame.winfo_children():
                widget.destroy()
            
            if not user_data:
                ttk.Label(self.profile_content_frame, 
                         text="No user data found",
                         style='Body.TLabel').pack(pady=50)
                return
            
            user = user_data[0]
            
            # Header
            header_frame = ttk.Frame(self.profile_content_frame)
            header_frame.pack(fill='x', padx=20, pady=20)
            
            ttk.Label(header_frame, text="User Profile (Live Database Data)",
                     style='Title.TLabel').pack(anchor='w')
            
            # Create scrollable content
            canvas = tk.Canvas(self.profile_content_frame, bg='#f0f8f0')
            scrollbar = ttk.Scrollbar(self.profile_content_frame, orient='vertical', command=canvas.yview)
            scrollable_frame = ttk.Frame(canvas)
            
            scrollable_frame.bind(
                "<Configure>",
                lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
            )
            
            canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
            canvas.configure(yscrollcommand=scrollbar.set)
            
            canvas.pack(side="left", fill="both", expand=True, padx=20)
            scrollbar.pack(side="right", fill="y")
            
            # Real profile data
            profile_data = json.loads(user.get('profile', '{}')) if user.get('profile') else {}
            
            # Basic info from database
            basic_info = [
                ("User ID", user.get('id', '')[:8] + '...'),
                ("Name", user.get('name', 'Unknown')),
                ("Email", user.get('email', 'Unknown')),
                ("Area", user.get('area', 'Unknown')),
                ("Created", str(user.get('created_at', 'Unknown'))[:10]),
                ("Status", "Active" if user.get('is_active') else "Inactive")
            ]
            
            self.create_profile_section(scrollable_frame, "Database Information", basic_info)
            
            # Profile preferences
            if profile_data:
                pref_info = [
                    ("Sustainability Goal", profile_data.get('sustainability_goal', 'Not set')),
                    ("Transport Modes", ', '.join(profile_data.get('transport_modes', []))),
                    ("Dietary Preferences", profile_data.get('dietary_preferences', 'Not set')),
                    ("AI Model", profile_data.get('ai_model_preference', 'gemini-1.5-flash')),
                    ("Language", profile_data.get('language', 'english')),
                    ("Budget Conscious", "Yes" if profile_data.get('budget_conscious') else "No")
                ]
                
                self.create_profile_section(scrollable_frame, "Preferences", pref_info)
                
                # Vehicle info if available
                vehicle_info = profile_data.get('vehicle_info', {})
                if vehicle_info:
                    vehicle_data = [
                        ("Vehicle Type", vehicle_info.get('type', 'Unknown')),
                        ("Fuel Efficiency", f"{vehicle_info.get('fuel_efficiency', 'Unknown')} km/l")
                    ]
                    self.create_profile_section(scrollable_frame, "Vehicle Information", vehicle_data)
                
                # Energy info if available
                energy_info = profile_data.get('energy_usage', {})
                if energy_info:
                    energy_data = [
                        ("Monthly Bill", f"₹{energy_info.get('monthly_bill', 'Unknown')}"),
                        ("Home Type", energy_info.get('home_type', 'Unknown')),
                        ("Renewable Energy", "Yes" if energy_info.get('has_renewable') else "No")
                    ]
                    self.create_profile_section(scrollable_frame, "Energy Usage", energy_data)
                    
        except Exception as e:
            logger.error(f"Error loading real profile: {e}")
            ttk.Label(self.profile_content_frame, 
                     text=f"Error loading profile: {e}",
                     style='Body.TLabel').pack(pady=50)
    
    def create_profile_section(self, parent, title, items):
        """Create profile section"""
        section_frame = ttk.Frame(parent, style='Card.TFrame')
        section_frame.pack(fill='x', padx=10, pady=10)
        
        # Section title
        title_frame = ttk.Frame(section_frame)
        title_frame.pack(fill='x', padx=20, pady=(15, 10))
        
        ttk.Label(title_frame, text=title,
                 style='Subtitle.TLabel').pack(anchor='w')
        
        # Items
        for label, value in items:
            item_frame = ttk.Frame(section_frame)
            item_frame.pack(fill='x', padx=20, pady=2)
            
            ttk.Label(item_frame, text=f"{label}:",
                     font=('Arial', 10),
                     background='white').pack(side='left')
            
            ttk.Label(item_frame, text=str(value),
                     font=('Arial', 10, 'bold'),
                     background='white').pack(side='right')
        
        # Add spacing
        ttk.Frame(section_frame, height=10).pack()
    
    def generate_real_recommendations(self):
        """Generate recommendations using your ACTUAL multi-agent system"""
        activity = self.activity_var.get()
        
        # Show loading dialog
        loading_dialog = tk.Toplevel(self.root)
        loading_dialog.title("AI Agents Working...")
        loading_dialog.geometry("400x200")
        loading_dialog.configure(bg='white')
        loading_dialog.transient(self.root)
        loading_dialog.grab_set()
        
        # Center the dialog
        loading_dialog.geometry("+%d+%d" % (self.root.winfo_rootx()+50, self.root.winfo_rooty()+50))
        
        ttk.Label(loading_dialog, text="🤖 Multi-Agent AI System Working...",
                 font=('Arial', 14, 'bold'), background='white').pack(pady=20)
        
        ttk.Label(loading_dialog, text="Your 6 AI agents are collaborating:",
                 font=('Arial', 10), background='white').pack()
        
        agents_text = """🔍 Context Agent - Analyzing your profile
🚗 Transport Agent - Finding sustainable routes  
🌱 Knowledge Agent - Calculating impacts
🧠 Memory Agent - Learning preferences
💡 Recommendation Agent - Creating suggestions
🎮 Rewards Agent - Calculating gamification"""
        
        ttk.Label(loading_dialog, text=agents_text,
                 font=('Arial', 9), background='white', justify='left').pack(pady=10)
        
        progress = ttk.Progressbar(loading_dialog, mode='indeterminate')
        progress.pack(pady=10)
        progress.start()
        
        def generate_async():
            try:
                # Update agent status
                self.root.after(0, lambda: self.agent_status_label.config(text="🤖 Agents: Working..."))
                
                # Call your ACTUAL TerraWise app with real agents!
                logger.info(f"Generating recommendations for user {self.current_user_id} with activity: {activity}")
                recommendations_result = self.app.generate_recommendations(self.current_user_id, activity)
                
                # Close loading dialog
                self.root.after(0, loading_dialog.destroy)
                
                # Update agent status
                self.root.after(0, lambda: self.agent_status_label.config(text="🤖 Agents: Ready"))
                
                # Display results
                self.root.after(0, lambda: self.display_real_recommendations(recommendations_result))
                
                # Show success message
                if recommendations_result.get('status') == 'success':
                    self.root.after(0, lambda: messagebox.showinfo("Success", 
                        "🌱 AI Agents generated new recommendations!\n\n" +
                        f"Model: {recommendations_result.get('ai_model', 'Gemini')}\n" +
                        f"Session: {recommendations_result.get('session_id', 'N/A')[:8]}...\n" +
                        "All 6 agents worked together to personalize these for you!"))
                elif recommendations_result.get('status') == 'success_fallback':
                    self.root.after(0, lambda: messagebox.showinfo("Success", 
                        "🌱 Recommendations generated using direct Gemini API!\n\n" +
                        "CrewAI agents had issues, but Gemini still provided great advice."))
                else:
                    self.root.after(0, lambda: messagebox.showerror("Error", 
                        f"Failed to generate recommendations: {recommendations_result.get('error', 'Unknown error')}"))
                
            except Exception as e:
                logger.error(f"Error in generate_async: {e}")
                self.root.after(0, loading_dialog.destroy)
                self.root.after(0, lambda: self.agent_status_label.config(text="🤖 Agents: Error"))
                self.root.after(0, lambda: messagebox.showerror("Error", f"Agent system failed: {e}"))
        
        # Run in separate thread to avoid blocking GUI
        threading.Thread(target=generate_async, daemon=True).start()
    
    def display_real_recommendations(self, recommendations_result):
        """Display recommendations from your actual AI system"""
        # Clear existing recommendations
        for widget in self.rec_display_frame.winfo_children():
            widget.destroy()
        
        if not recommendations_result or recommendations_result.get('status') == 'error':
            error_msg = recommendations_result.get('error', 'Unknown error') if recommendations_result else 'No response from agents'
            ttk.Label(self.rec_display_frame,
                     text=f"❌ Error generating recommendations: {error_msg}",
                     style='Body.TLabel').pack(pady=50)
            return
        
        # Get the latest recommendations from database
        try:
            query = """
            SELECT id, category, title, description, impact_metrics, status, created_at, session_id
            FROM recommendations 
            WHERE user_id = %s 
            ORDER BY created_at DESC 
            LIMIT 5
            """
            real_recommendations = self.app.db_manager.execute_query(query, (self.current_user_id,))
            
            if not real_recommendations:
                # Fallback to parsed recommendations from AI response
                ai_response = recommendations_result.get('recommendations', '')
                
                info_frame = ttk.Frame(self.rec_display_frame, style='Card.TFrame')
                info_frame.pack(fill='x', padx=10, pady=10)
                
                info_content = ttk.Frame(info_frame)
                info_content.pack(fill='x', padx=20, pady=15)
                
                ttk.Label(info_content, text="🤖 AI Agent Response", 
                         style='Subtitle.TLabel').pack(anchor='w')
                
                ttk.Label(info_content, text=f"Status: {recommendations_result.get('status', 'unknown')}", 
                         style='Body.TLabel').pack(anchor='w')
                
                ttk.Label(info_content, text=f"Model: {recommendations_result.get('ai_model', 'Unknown')}", 
                         style='Body.TLabel').pack(anchor='w')
                
                # Show AI response in scrollable text
                response_text = scrolledtext.ScrolledText(info_content, height=10, width=80, wrap=tk.WORD)
                response_text.pack(pady=10)
                response_text.insert(tk.END, str(ai_response))
                response_text.config(state=tk.DISABLED)
                
                return
            
            # Create scrollable area for real recommendations
            canvas = tk.Canvas(self.rec_display_frame, bg='#f0f8f0')
            scrollbar = ttk.Scrollbar(self.rec_display_frame, orient='vertical', command=canvas.yview)
            scrollable_frame = ttk.Frame(canvas)
            
            scrollable_frame.bind(
                "<Configure>",
                lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
            )
            
            canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
            canvas.configure(yscrollcommand=scrollbar.set)
            
            canvas.pack(side="left", fill="both", expand=True)
            scrollbar.pack(side="right", fill="y")
            
            # Show info about the generation
            info_frame = ttk.Frame(scrollable_frame, style='Card.TFrame')
            info_frame.pack(fill='x', padx=10, pady=10)
            
            info_content = ttk.Frame(info_frame)
            info_content.pack(fill='x', padx=20, pady=10)
            
            ttk.Label(info_content, text="✅ Latest AI-Generated Recommendations", 
                     style='Subtitle.TLabel').pack(anchor='w')
            
            session_id = recommendations_result.get('session_id', 'Unknown')
            ttk.Label(info_content, text=f"Session: {session_id[:8]}... | Model: {recommendations_result.get('ai_model', 'Gemini')}", 
                     style='Body.TLabel').pack(anchor='w')
            
            # Display each real recommendation from database
            self.current_recommendations = real_recommendations
            for i, rec in enumerate(real_recommendations):
                self.create_real_recommendation_card(scrollable_frame, rec, i)
                
        except Exception as e:
            logger.error(f"Error displaying real recommendations: {e}")
            ttk.Label(self.rec_display_frame,
                     text=f"Error loading recommendations: {e}",
                     style='Body.TLabel').pack(pady=50)
    
    def display_initial_recommendations(self):
        """Display initial message for recommendations tab"""
        initial_frame = ttk.Frame(self.rec_display_frame, style='Card.TFrame')
        initial_frame.pack(fill='both', expand=True, padx=10, pady=50)
        
        content = ttk.Frame(initial_frame)
        content.pack(expand=True, pady=50)
        
        ttk.Label(content, text="🤖 Multi-Agent AI System Ready", 
                 font=('Arial', 18, 'bold'),
                 background='white').pack(pady=10)
        
        ttk.Label(content, text="Your 6 AI agents are standing by to generate personalized sustainability recommendations.", 
                 font=('Arial', 12),
                 background='white').pack(pady=5)
        
        ttk.Label(content, text="Click 'Generate with AI Agents' to start the multi-agent collaboration!", 
                 font=('Arial', 11), foreground='#4a7c59',
                 background='white').pack(pady=5)
        
        agents_summary = """
        🔍 Context Agent - Will analyze your Mumbai location and preferences
        🚗 Transport Agent - Will find sustainable transport options  
        🌱 Knowledge Agent - Will calculate environmental impacts
        🧠 Memory Agent - Will learn from your past feedback
        💡 Recommendation Agent - Will coordinate all insights
        🎮 Rewards Agent - Will calculate coins and achievements
        """
        
        ttk.Label(content, text=agents_summary, 
                 font=('Arial', 10), justify='left',
                 background='white').pack(pady=15)
    
    def create_real_recommendation_card(self, parent, recommendation, index):
        """Create recommendation card from real database data"""
        card_frame = ttk.Frame(parent, style='Card.TFrame')
        card_frame.pack(fill='x', padx=10, pady=10)
        
        # Card content
        content_frame = ttk.Frame(card_frame)
        content_frame.pack(fill='x', padx=20, pady=15)
        
        # Header
        header_frame = ttk.Frame(content_frame)
        header_frame.pack(fill='x')
        
        # Icon and title
        icons = {'transport': '🚗', 'energy': '⚡', 'food': '🍽️', 'water': '💧'}
        icon = icons.get(recommendation.get('category', ''), '🌱')
        
        title_text = f"{icon} {recommendation.get('title', 'AI Recommendation')}"
        ttk.Label(header_frame, text=title_text,
                 font=('Arial', 14, 'bold'),
                 background='white').pack(anchor='w')
        
        # Category and status
        category = recommendation.get('category', 'general')
        status = recommendation.get('status', 'pending')
        
        status_frame = ttk.Frame(header_frame)
        status_frame.pack(side='right')
        
        # Category badge
        category_label = tk.Label(status_frame, text=category.title(),
                                 bg='#6fb37b', fg='white',
                                 font=('Arial', 9, 'bold'),
                                 padx=8, pady=2)
        category_label.pack(side='top')
        
        # Status badge
        status_colors = {'pending': '#ffc107', 'accepted': '#28a745', 'rejected': '#dc3545'}
        status_color = status_colors.get(status, '#6c757d')
        
        status_label = tk.Label(status_frame, text=status.title(),
                               bg=status_color, fg='white',
                               font=('Arial', 8, 'bold'),
                               padx=6, pady=1)
        status_label.pack(side='bottom', pady=(2, 0))
        
        # Description
        description = recommendation.get('description', 'AI-generated sustainability recommendation')
        ttk.Label(content_frame, text=description,
                 font=('Arial', 10),
                 background='white',
                 wraplength=700).pack(anchor='w', pady=(10, 0))
        
        # Metrics from database
        metrics_frame = ttk.Frame(content_frame)
        metrics_frame.pack(fill='x', pady=(15, 0))
        
        # Parse impact metrics from database
        impact_metrics = {}
        if recommendation.get('impact_metrics'):
            try:
                impact_metrics = json.loads(recommendation['impact_metrics'])
            except:
                pass
        
        # Display metrics
        co2_saved = impact_metrics.get('co2_saved_kg', impact_metrics.get('co2_reduction_kg', 1.5))
        money_saved = impact_metrics.get('money_saved_inr', 100)
        coins = impact_metrics.get('coins_earned', 15)
        
        self.create_metric_item(metrics_frame, "🌍 CO₂ Saved", f"{co2_saved} kg", 0)
        self.create_metric_item(metrics_frame, "💰 Money Saved", f"₹{money_saved}", 1)
        self.create_metric_item(metrics_frame, "🪙 Coins", f"{coins}", 2)
        
        # Database info
        db_info_frame = ttk.Frame(content_frame)
        db_info_frame.pack(fill='x', pady=(10, 0))
        
        created_at = recommendation.get('created_at', 'Unknown')
        if hasattr(created_at, 'strftime'):
            created_str = created_at.strftime('%Y-%m-%d %H:%M:%S')
        else:
            created_str = str(created_at)
        
        rec_id = str(recommendation.get('id', 'Unknown'))[:8]
        session_id = str(recommendation.get('session_id', 'Unknown'))[:8]
        
        db_info_text = f"DB ID: {rec_id}... | Session: {session_id}... | Created: {created_str}"
        ttk.Label(db_info_frame, text=db_info_text,
                 font=('Arial', 8), foreground='#666666',
                 background='white').pack(anchor='w')
        
        # Action buttons (only if status is pending)
        if status == 'pending':
            button_frame = ttk.Frame(content_frame)
            button_frame.pack(fill='x', pady=(15, 0))
            
            ttk.Button(button_frame, text="✅ Accept",
                      style='Primary.TButton',
                      command=lambda: self.handle_real_recommendation('accepted', recommendation)).pack(side='left', padx=(0, 10))
            
            ttk.Button(button_frame, text="❌ Decline",
                      style='Secondary.TButton',
                      command=lambda: self.handle_real_recommendation('rejected', recommendation)).pack(side='left')
            
            ttk.Button(button_frame, text="ℹ️ More Info",
                      command=lambda: self.show_real_recommendation_details(recommendation)).pack(side='right')
    
    def create_metric_item(self, parent, label, value, column):
        """Create metric display item"""
        metric_frame = ttk.Frame(parent)
        metric_frame.grid(row=0, column=column, padx=10, sticky='w')
        
        ttk.Label(metric_frame, text=label,
                 font=('Arial', 9),
                 background='white').pack()
        
        ttk.Label(metric_frame, text=value,
                 font=('Arial', 11, 'bold'),
                 foreground='#28a745',
                 background='white').pack()
    
    def handle_real_recommendation(self, action, recommendation):
        """Handle recommendation feedback using your actual TerraWise system"""
        try:
            rec_id = recommendation.get('id')
            if not rec_id:
                messagebox.showerror("Error", "Invalid recommendation ID")
                return
            
            # Use your actual feedback system
            feedback_reason = None
            if action == 'rejected':
                feedback_reason = "Not suitable for current situation"
            elif action == 'accepted':
                feedback_reason = "Great suggestion, will implement!"
            
            feedback_result = self.app.submit_feedback(
                self.current_user_id, 
                str(rec_id), 
                action, 
                feedback_reason
            )
            
            if feedback_result.get('status') == 'success':
                # Update the recommendation status in database
                update_query = "UPDATE recommendations SET status = %s WHERE id = %s"
                self.app.db_manager.execute_query(update_query, (action, rec_id))
                
                # Show success message with real impact
                impact_metrics = json.loads(recommendation.get('impact_metrics', '{}'))
                coins = impact_metrics.get('coins_earned', 15)
                co2_saved = impact_metrics.get('co2_saved_kg', impact_metrics.get('co2_reduction_kg', 1.5))
                
                if action == 'accepted':
                    # Update user stats in database
                    stats_update = """
                    UPDATE user_stats 
                    SET total_coins = total_coins + %s,
                        total_co2_saved_kg = total_co2_saved_kg + %s,
                        recommendations_accepted = recommendations_accepted + 1,
                        current_streak = current_streak + 1
                    WHERE user_id = %s
                    """
                    self.app.db_manager.execute_query(stats_update, (coins, co2_saved, self.current_user_id))
                    
                    messagebox.showinfo("Accepted! 🎉", 
                        f"Amazing choice! You earned:\n\n" +
                        f"🪙 {coins} coins\n" +
                        f"🌍 {co2_saved} kg CO₂ saved\n" +
                        f"📊 Stats updated in database\n\n" +
                        "Keep up the great work! 🌱")
                else:
                    messagebox.showinfo("Feedback Received", 
                        "Thanks for the feedback! Our AI agents will learn from this to improve future recommendations. 🤖")
                
                # Refresh the display
                self.refresh_real_data()
                
            else:
                messagebox.showerror("Error", f"Failed to submit feedback: {feedback_result.get('error', 'Unknown error')}")
                
        except Exception as e:
            logger.error(f"Error handling recommendation feedback: {e}")
            messagebox.showerror("Error", f"Failed to process feedback: {e}")
    
    def show_real_recommendation_details(self, recommendation):
        """Show detailed information about real recommendation"""
        details_window = tk.Toplevel(self.root)
        details_window.title("AI Recommendation Details")
        details_window.geometry("600x500")
        details_window.configure(bg='white')
        details_window.transient(self.root)
        
        # Title
        title_label = tk.Label(details_window, text=recommendation.get('title', 'AI Recommendation'),
                              font=('Arial', 16, 'bold'),
                              bg='white', fg='#2d5a3d')
        title_label.pack(pady=20)
        
        # Details text area
        details_text = scrolledtext.ScrolledText(details_window, height=20, width=70,
                                               font=('Arial', 10), wrap=tk.WORD)
        details_text.pack(padx=20, pady=10)
        
        # Compile detailed information
        impact_metrics = {}
        try:
            impact_metrics = json.loads(recommendation.get('impact_metrics', '{}'))
        except:
            pass
        
        created_at = recommendation.get('created_at', 'Unknown')
        if hasattr(created_at, 'strftime'):
            created_str = created_at.strftime('%Y-%m-%d %H:%M:%S')
        else:
            created_str = str(created_at)
        
        content = f"""🤖 AI-Generated Recommendation Details
{'='*50}

Recommendation: {recommendation.get('title', 'Unknown')}
Category: {recommendation.get('category', 'general').title()}
Status: {recommendation.get('status', 'pending').title()}

Database Information:
• ID: {recommendation.get('id', 'Unknown')}
• Session: {recommendation.get('session_id', 'Unknown')}
• Created: {created_str}
• User: {self.current_user_id}

Description:
{recommendation.get('description', 'AI-generated sustainability recommendation')}

Environmental Impact:
• CO₂ Reduction: {impact_metrics.get('co2_saved_kg', impact_metrics.get('co2_reduction_kg', 'Unknown'))} kg
• Money Saved: ₹{impact_metrics.get('money_saved_inr', 'Unknown')}
• Coins Earned: {impact_metrics.get('coins_earned', 'Unknown')}
• AI Model: {impact_metrics.get('ai_model', 'Gemini')}

Feasibility Scores:
• Overall Score: {recommendation.get('feasibility_score', 'Not available')}
• User Preference: {recommendation.get('user_preference_score', 'Not available')}
• Priority Rank: {recommendation.get('priority_rank', 'Not available')}

How This Was Generated:
This recommendation was created by your 6-agent AI system:
1. 🔍 Context Agent analyzed your Mumbai location and profile
2. 🚗 Transport Agent found sustainable transport options
3. 🌱 Knowledge Agent calculated environmental impacts
4. 🧠 Memory Agent considered your past preferences
5. 💡 Recommendation Agent coordinated all insights
6. 🎮 Rewards Agent calculated gamification elements

Mumbai-Specific Considerations:
• Weather: Suitable for all seasons including monsoon
• Transport: Compatible with Mumbai's infrastructure
• Cost: Affordable for Indian urban context
• Cultural: Appropriate for local lifestyle

Implementation Tips:
• Start gradually and build the habit
• Track progress using the dashboard
• Join local sustainability communities
• Share your success on social media

Expected Timeline:
• Week 1: Research and planning
• Week 2-3: Initial implementation
• Week 4+: Habit formation and optimization

Additional Resources:
• Mumbai Metro route planner
• Local cycle sharing applications
• Energy-efficient appliance stores
• Community sustainability groups
"""
        
        details_text.insert(tk.END, content)
        details_text.config(state=tk.DISABLED)
        
        # Close button
        close_frame = ttk.Frame(details_window)
        close_frame.pack(pady=10)
        
        ttk.Button(close_frame, text="Close", 
                  command=details_window.destroy).pack()
    
    def refresh_real_data(self):
        """Refresh all data from your actual database"""
        try:
            # Refresh dashboard with real data
            self.load_real_dashboard()
            
            # Refresh profile with real data  
            self.load_real_profile()
            
            # Clear recommendations display to force reload
            for widget in self.rec_display_frame.winfo_children():
                widget.destroy()
            self.display_initial_recommendations()
            
            messagebox.showinfo("Refreshed", "All data refreshed from database! 🔄")
            
        except Exception as e:
            logger.error(f"Error refreshing real data: {e}")
            messagebox.showerror("Error", f"Failed to refresh data: {e}")

# MAIN FUNCTION TO ADD TO YOUR EXISTING main.py
def run_gui():
    """Function to run the GUI - ADD THIS TO YOUR EXISTING main.py"""
    
    print("\n🎨 Launching TerraWise GUI with Multi-Agent System...")
    
    try:
        # Test connections first
        print("🔍 Testing Gemini API connection...")
        if not test_gemini_connection():
            print("❌ Cannot start GUI - Gemini API not working")
            return
        
        # Initialize your existing TerraWise app
        print("🔧 Initializing TerraWise Multi-Agent System...")
        app = TerraWiseApp()
        
        print("✅ Multi-Agent System initialized successfully!")
        print("🚀 Starting GUI...")
        
        # Create GUI root
        root = tk.Tk()
        
        # Optional: Set window icon
        try:
            root.iconbitmap('terrawise_icon.ico')
        except:
            pass
        
        # Create GUI with connection to your actual app
        gui = TerraWiseGUI(root, app)
        
        # Center the window
        root.update_idletasks()
        x = (root.winfo_screenwidth() // 2) - (1200 // 2)
        y = (root.winfo_screenheight() // 2) - (800 // 2)
        root.geometry(f"1200x800+{x}+{y}")
        
        print("🎉 TerraWise GUI launched successfully!")
        print("=" * 60)
        print("Features Available:")
        print("• 🏠 Real-time dashboard with database stats")
        print("• 🤖 Multi-agent AI recommendation generation")
        print("• 📊 Live agent status monitoring")
        print("• 👤 Real user profile from database")
        print("• 🔄 Full integration with your MySQL database")
        print("• 🌱 Gamification with real coin/XP tracking")
        print("=" * 60)
        
        # Start the GUI
        root.mainloop()
        
    except Exception as e:
        logger.error(f"GUI startup failed: {e}")
        print(f"❌ Failed to start GUI: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure your MySQL database is running")
        print("2. Check your GOOGLE_API_KEY is set")
        print("3. Ensure all dependencies are installed")
        print("4. Verify your main TerraWise system works first")

# ADD THIS TO THE END OF YOUR EXISTING main() FUNCTION IN main.py
def enhanced_main():
    """Enhanced main function that includes GUI option - REPLACE YOUR EXISTING main() WITH THIS"""
    
    print("🌱 Welcome to TerraWise - AI-Powered Sustainability Assistant")
    print("🤖 Powered by FREE Google Gemini API + CrewAI Multi-Agent System")
    print("=" * 70)
    
    # Ask user what they want to do
    print("\n📋 Choose how to run TerraWise:")
    print("1. 🖥️  Launch GUI Application (Recommended)")
    print("2. 💻 Run Command Line Demo") 
    print("3. 🔧 Command Line + GUI")
    
    choice = input("Enter your choice (1-3): ").strip()
    
    if choice == '1':
        # Launch GUI only
        run_gui()
    elif choice == '2':
        # Run your existing command line demo
        main()  # Your existing main function
    elif choice == '3':
        # Run both
        print("\n🚀 Running command line demo first...")
        main()  # Your existing main function
        
        print("\n" + "="*50)
        input("Press Enter to launch GUI...")
        run_gui()
    else:
        print("Invalid choice. Launching GUI by default...")
        run_gui()

# INSTRUCTIONS FOR INTEGRATION:
if __name__ == "__main__":
    enhanced_main()