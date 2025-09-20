"""
TerraWise GUI Application with Rate Limiting
Enhanced GUI for TerraWise sustainability system with proper Gemini API rate limiting
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import json
import threading
import time
import queue
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging
from dataclasses import dataclass
import asyncio
from concurrent.futures import ThreadPoolExecutor
import sys
import os

# Rate limiting imports
from collections import deque
import functools

# Import your TerraWise components
# Note: Make sure your TerraWise code is available
try:
    from main import TerraWiseApp, DatabaseManager, setup_demo_user
except ImportError:
    print("Please ensure the TerraWise module is in your Python path")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===========================
# RATE LIMITER CLASS
# ===========================

class GeminiRateLimiter:
    """Rate limiter specifically for Gemini API to avoid 15 requests/minute limit"""
    
    def __init__(self, max_requests=12, time_window=60):  # Conservative: 12 requests per minute
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = deque()
        self.lock = threading.Lock()
    
    def can_make_request(self) -> bool:
        """Check if we can make a request without hitting rate limit"""
        with self.lock:
            now = time.time()
            # Remove old requests outside the time window
            while self.requests and self.requests[0] <= now - self.time_window:
                self.requests.popleft()
            
            return len(self.requests) < self.max_requests
    
    def wait_time_until_request(self) -> float:
        """Return how long to wait before making next request"""
        with self.lock:
            now = time.time()
            # Remove old requests
            while self.requests and self.requests[0] <= now - self.time_window:
                self.requests.popleft()
            
            if len(self.requests) < self.max_requests:
                return 0.0
            
            # Calculate wait time until oldest request expires
            oldest_request = self.requests[0]
            wait_time = (oldest_request + self.time_window) - now
            return max(0.0, wait_time)
    
    def record_request(self):
        """Record that a request was made"""
        with self.lock:
            self.requests.append(time.time())
    
    def rate_limited_call(self, func, *args, **kwargs):
        """Make a rate-limited function call"""
        wait_time = self.wait_time_until_request()
        if wait_time > 0:
            logger.info(f"Rate limiting: waiting {wait_time:.1f} seconds")
            time.sleep(wait_time)
        
        self.record_request()
        return func(*args, **kwargs)

# Global rate limiter instance
rate_limiter = GeminiRateLimiter()

# ===========================
# USER MANAGEMENT
# ===========================

@dataclass
class User:
    id: str
    name: str
    email: str
    area: str
    profile: Dict
    created_at: str
    total_coins: int = 0
    total_co2_saved: float = 0.0
    current_streak: int = 0

class UserManager:
    """Manage users from database"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
    
    def get_all_users(self) -> List[User]:
        """Get all active users from database"""
        query = """
        SELECT u.id, u.name, u.email, u.area, u.profile, u.created_at,
               COALESCE(us.total_coins, 0) as total_coins,
               COALESCE(us.total_co2_saved_kg, 0) as total_co2_saved,
               COALESCE(us.current_streak, 0) as current_streak
        FROM users u
        LEFT JOIN user_stats us ON u.id = us.user_id
        WHERE u.is_active = TRUE
        ORDER BY u.created_at DESC
        """
        
        users_data = self.db_manager.execute_query(query)
        users = []
        
        for user_data in users_data:
            profile = json.loads(user_data['profile']) if user_data['profile'] else {}
            user = User(
                id=user_data['id'],
                name=user_data['name'],
                email=user_data['email'],
                area=user_data['area'],
                profile=profile,
                created_at=str(user_data['created_at']),
                total_coins=user_data['total_coins'],
                total_co2_saved=float(user_data['total_co2_saved']),
                current_streak=user_data['current_streak']
            )
            users.append(user)
        
        return users
    
    def create_user(self, name: str, email: str, area: str, profile: Dict) -> str:
        """Create a new user"""
        import uuid
        user_id = str(uuid.uuid4())
        
        query = """
        INSERT INTO users (id, name, email, area, profile, is_active, created_at) 
        VALUES (%s, %s, %s, %s, %s, TRUE, NOW())
        """
        
        profile_json = json.dumps(profile)
        self.db_manager.execute_query(query, (user_id, name, email, area, profile_json))
        
        return user_id

# ===========================
# MAIN GUI APPLICATION
# ===========================

class TerraWiseGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("TerraWise - AI Sustainability Assistant")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f8ff')
        
        # Initialize components
        self.db_manager = DatabaseManager()
        self.user_manager = UserManager(self.db_manager)
        
        # Rate-limited TerraWise app
        self.terrawise_app = TerraWiseApp()
        
        # Threading for API calls
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.result_queue = queue.Queue()
        
        # Current user
        self.current_user = None
        self.users_list = []
        
        # Setup GUI
        self.setup_styles()
        self.create_widgets()
        self.load_users()
        
        # Start result processor
        self.process_results()
    
    def setup_styles(self):
        """Setup custom styles for the GUI"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Custom styles
        style.configure('Title.TLabel', font=('Arial', 16, 'bold'), background='#f0f8ff')
        style.configure('Header.TLabel', font=('Arial', 12, 'bold'), background='#f0f8ff')
        style.configure('Info.TLabel', font=('Arial', 10), background='#f0f8ff')
        style.configure('Success.TLabel', foreground='#28a745', font=('Arial', 10, 'bold'))
        style.configure('Warning.TLabel', foreground='#ffc107', font=('Arial', 10, 'bold'))
        style.configure('Error.TLabel', foreground='#dc3545', font=('Arial', 10, 'bold'))
    
    def create_widgets(self):
        """Create all GUI widgets"""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="🌱 TerraWise - AI Sustainability Assistant", 
                               style='Title.TLabel')
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Left Panel - User Selection
        self.create_user_panel(main_frame)
        
        # Right Panel - User Details and Actions
        self.create_user_details_panel(main_frame)
        
        # Bottom Panel - Results
        self.create_results_panel(main_frame)
        
        # Rate limiter status
        self.create_status_panel(main_frame)
    
    def create_user_panel(self, parent):
        """Create user selection panel"""
        user_frame = ttk.LabelFrame(parent, text="👥 Select User", padding="10")
        user_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # User listbox
        self.user_listbox = tk.Listbox(user_frame, height=15, font=('Arial', 10))
        self.user_listbox.pack(fill=tk.BOTH, expand=True)
        self.user_listbox.bind('<<ListboxSelect>>', self.on_user_select)
        
        # Buttons
        button_frame = ttk.Frame(user_frame)
        button_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Button(button_frame, text="🔄 Refresh Users", 
                  command=self.load_users).pack(side=tk.LEFT, padx=(0, 5))
        
        ttk.Button(button_frame, text="➕ New User", 
                  command=self.show_new_user_dialog).pack(side=tk.LEFT)
    
    def create_user_details_panel(self, parent):
        """Create user details and actions panel"""
        details_frame = ttk.LabelFrame(parent, text="👤 User Details & Actions", padding="10")
        details_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        details_frame.columnconfigure(0, weight=1)
        details_frame.rowconfigure(3, weight=1)
        
        # User info
        self.user_info_frame = ttk.Frame(details_frame)
        self.user_info_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 15))
        
        # Activity selection
        activity_frame = ttk.LabelFrame(details_frame, text="🎯 Current Activity")
        activity_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 15))
        
        self.activity_var = tk.StringVar(value="general")
        activities = [
            ("🏠 At Home", "at_home"),
            ("🚗 Traveling to Work", "traveling_to_work"),
            ("🛒 Shopping", "shopping"),
            ("🍳 Cooking", "cooking"),
            ("💼 At Office", "at_office"),
            ("🌐 General", "general")
        ]
        
        for i, (text, value) in enumerate(activities):
            ttk.Radiobutton(activity_frame, text=text, variable=self.activity_var, 
                           value=value).grid(row=i//2, column=i%2, sticky=tk.W, padx=5, pady=2)
        
        # Action buttons
        action_frame = ttk.Frame(details_frame)
        action_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 15))
        
        self.recommend_btn = ttk.Button(action_frame, text="🤖 Generate AI Recommendations", 
                                       command=self.generate_recommendations, state='disabled')
        self.recommend_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        self.dashboard_btn = ttk.Button(action_frame, text="📊 View Dashboard", 
                                       command=self.view_dashboard, state='disabled')
        self.dashboard_btn.pack(side=tk.LEFT)
        
        # User statistics
        self.stats_frame = ttk.LabelFrame(details_frame, text="📈 User Statistics")
        self.stats_frame.grid(row=3, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
    
    def create_results_panel(self, parent):
        """Create results display panel"""
        results_frame = ttk.LabelFrame(parent, text="📋 AI Results", padding="10")
        results_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), 
                          pady=(15, 0))
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)
        
        # Results text area
        self.results_text = scrolledtext.ScrolledText(results_frame, height=15, 
                                                     font=('Consolas', 10), wrap=tk.WORD)
        self.results_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
    
    def create_status_panel(self, parent):
        """Create status panel for rate limiting info"""
        status_frame = ttk.Frame(parent)
        status_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        
        self.status_label = ttk.Label(status_frame, text="🟢 Ready - Gemini API Rate Limiter Active", 
                                     style='Success.TLabel')
        self.status_label.pack(side=tk.LEFT)
        
        # API usage info
        self.api_usage_label = ttk.Label(status_frame, text="API Calls: 0/12 per minute", 
                                        style='Info.TLabel')
        self.api_usage_label.pack(side=tk.RIGHT)
    
    def load_users(self):
        """Load users from database"""
        try:
            self.users_list = self.user_manager.get_all_users()
            self.user_listbox.delete(0, tk.END)
            
            if not self.users_list:
                self.user_listbox.insert(tk.END, "No users found - Create a new user")
                return
            
            for user in self.users_list:
                display_text = f"{user.name} ({user.area}) - {user.total_coins} coins"
                self.user_listbox.insert(tk.END, display_text)
            
            self.update_status("✅ Users loaded successfully", "success")
            
        except Exception as e:
            logger.error(f"Error loading users: {e}")
            messagebox.showerror("Error", f"Failed to load users: {e}")
    
    def on_user_select(self, event):
        """Handle user selection"""
        selection = event.widget.curselection()
        if not selection or not self.users_list:
            return
        
        index = selection[0]
        if index < len(self.users_list):
            self.current_user = self.users_list[index]
            self.display_user_info()
            self.recommend_btn.config(state='normal')
            self.dashboard_btn.config(state='normal')
    
    def display_user_info(self):
        """Display selected user information"""
        if not self.current_user:
            return
        
        # Clear previous info
        for widget in self.user_info_frame.winfo_children():
            widget.destroy()
        
        # Display user info
        ttk.Label(self.user_info_frame, text=f"👤 {self.current_user.name}", 
                 style='Header.TLabel').grid(row=0, column=0, sticky=tk.W, columnspan=2)
        
        ttk.Label(self.user_info_frame, text=f"📧 {self.current_user.email}", 
                 style='Info.TLabel').grid(row=1, column=0, sticky=tk.W, columnspan=2)
        
        ttk.Label(self.user_info_frame, text=f"📍 {self.current_user.area}", 
                 style='Info.TLabel').grid(row=2, column=0, sticky=tk.W, columnspan=2)
        
        # Display stats in stats frame
        for widget in self.stats_frame.winfo_children():
            widget.destroy()
        
        ttk.Label(self.stats_frame, text=f"🪙 Coins: {self.current_user.total_coins}", 
                 style='Info.TLabel').pack(anchor=tk.W)
        
        ttk.Label(self.stats_frame, text=f"🌱 CO2 Saved: {self.current_user.total_co2_saved:.2f} kg", 
                 style='Info.TLabel').pack(anchor=tk.W)
        
        ttk.Label(self.stats_frame, text=f"🔥 Streak: {self.current_user.current_streak} days", 
                 style='Info.TLabel').pack(anchor=tk.W)
        
        # Display profile info
        if self.current_user.profile:
            profile_text = "\n👤 Profile:\n"
            for key, value in self.current_user.profile.items():
                if isinstance(value, dict):
                    profile_text += f"  • {key}: {json.dumps(value, indent=2)}\n"
                else:
                    profile_text += f"  • {key}: {value}\n"
            
            profile_label = ttk.Label(self.stats_frame, text=profile_text, 
                                    style='Info.TLabel', justify=tk.LEFT)
            profile_label.pack(anchor=tk.W, pady=(10, 0))
    
    def generate_recommendations(self):
        """Generate AI recommendations with rate limiting"""
        if not self.current_user:
            messagebox.showwarning("Warning", "Please select a user first")
            return
        
        # Check rate limit
        if not rate_limiter.can_make_request():
            wait_time = rate_limiter.wait_time_until_request()
            self.update_status(f"⏱️ Rate limited - waiting {wait_time:.1f}s", "warning")
            messagebox.showwarning("Rate Limited", 
                                 f"Please wait {wait_time:.1f} seconds before making another request\n"
                                 "Gemini API allows only 15 requests per minute")
            return
        
        activity = self.activity_var.get()
        self.recommend_btn.config(state='disabled', text="🤖 Generating...")
        self.update_status("🔄 Generating recommendations...", "info")
        
        # Submit to thread pool
        future = self.executor.submit(self.generate_recommendations_worker, 
                                    self.current_user.id, activity)
        
        # Check result periodically
        self.root.after(100, lambda: self.check_recommendation_result(future))
    
    def generate_recommendations_worker(self, user_id: str, activity: str):
        """Worker function for generating recommendations with rate limiting"""
        try:
            # Use rate limiter
            result = rate_limiter.rate_limited_call(
                self.terrawise_app.generate_recommendations,
                user_id, activity
            )
            
            # Update API usage count
            current_requests = len(rate_limiter.requests)
            self.result_queue.put(('api_usage', current_requests))
            
            return {'status': 'success', 'data': result}
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def check_recommendation_result(self, future):
        """Check if recommendation generation is complete"""
        if future.done():
            try:
                result = future.result()
                self.result_queue.put(('recommendations', result))
            except Exception as e:
                self.result_queue.put(('error', f"Recommendation generation failed: {e}"))
            
            self.recommend_btn.config(state='normal', text="🤖 Generate AI Recommendations")
        else:
            self.root.after(100, lambda: self.check_recommendation_result(future))
    
    def view_dashboard(self):
        """View user dashboard"""
        if not self.current_user:
            messagebox.showwarning("Warning", "Please select a user first")
            return
        
        self.dashboard_btn.config(state='disabled', text="📊 Loading...")
        self.update_status("🔄 Loading dashboard...", "info")
        
        # Submit to thread pool
        future = self.executor.submit(self.view_dashboard_worker, self.current_user.id)
        
        # Check result periodically
        self.root.after(100, lambda: self.check_dashboard_result(future))
    
    def view_dashboard_worker(self, user_id: str):
        """Worker function for getting dashboard"""
        try:
            result = self.terrawise_app.get_user_dashboard(user_id)
            return {'status': 'success', 'data': result}
        except Exception as e:
            logger.error(f"Error getting dashboard: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def check_dashboard_result(self, future):
        """Check if dashboard loading is complete"""
        if future.done():
            try:
                result = future.result()
                self.result_queue.put(('dashboard', result))
            except Exception as e:
                self.result_queue.put(('error', f"Dashboard loading failed: {e}"))
            
            self.dashboard_btn.config(state='normal', text="📊 View Dashboard")
        else:
            self.root.after(100, lambda: self.check_dashboard_result(future))
    
    def show_new_user_dialog(self):
        """Show dialog to create new user"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Create New User")
        dialog.geometry("400x500")
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Center the dialog
        dialog.geometry("+%d+%d" % (self.root.winfo_rootx() + 50, self.root.winfo_rooty() + 50))
        
        # Form fields
        ttk.Label(dialog, text="Create New User", style='Header.TLabel').pack(pady=10)
        
        # Name
        ttk.Label(dialog, text="Name:").pack(anchor=tk.W, padx=20)
        name_var = tk.StringVar()
        ttk.Entry(dialog, textvariable=name_var, width=40).pack(padx=20, pady=(0, 10))
        
        # Email
        ttk.Label(dialog, text="Email:").pack(anchor=tk.W, padx=20)
        email_var = tk.StringVar()
        ttk.Entry(dialog, textvariable=email_var, width=40).pack(padx=20, pady=(0, 10))
        
        # Area
        ttk.Label(dialog, text="Area/Location:").pack(anchor=tk.W, padx=20)
        area_var = tk.StringVar()
        area_combo = ttk.Combobox(dialog, textvariable=area_var, width=37)
        area_combo['values'] = ('Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Kolkata', 'Pune', 'Hyderabad')
        area_combo.pack(padx=20, pady=(0, 10))
        
        # Sustainability Goal
        ttk.Label(dialog, text="Sustainability Goal:").pack(anchor=tk.W, padx=20)
        goal_var = tk.StringVar()
        goal_combo = ttk.Combobox(dialog, textvariable=goal_var, width=37)
        goal_combo['values'] = ('reduce_carbon_footprint', 'save_money', 'live_healthier', 'help_environment')
        goal_combo.pack(padx=20, pady=(0, 10))
        
        # Transport modes
        ttk.Label(dialog, text="Available Transport Modes:").pack(anchor=tk.W, padx=20)
        transport_frame = ttk.Frame(dialog)
        transport_frame.pack(padx=20, pady=(0, 10))
        
        transport_vars = {}
        transport_modes = ['car', 'metro', 'bus', 'auto', 'bike', 'walk', 'cycle']
        for i, mode in enumerate(transport_modes):
            var = tk.BooleanVar()
            transport_vars[mode] = var
            ttk.Checkbutton(transport_frame, text=mode.title(), variable=var).grid(
                row=i//3, column=i%3, sticky=tk.W, padx=5)
        
        # Budget conscious
        budget_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(dialog, text="Budget Conscious", variable=budget_var).pack(anchor=tk.W, padx=20, pady=5)
        
        # Buttons
        button_frame = ttk.Frame(dialog)
        button_frame.pack(pady=20)
        
        def create_user():
            try:
                # Validate input
                if not all([name_var.get(), email_var.get(), area_var.get()]):
                    messagebox.showwarning("Warning", "Please fill all required fields")
                    return
                
                # Build profile
                profile = {
                    'sustainability_goal': goal_var.get() or 'reduce_carbon_footprint',
                    'transport_modes': [mode for mode, var in transport_vars.items() if var.get()],
                    'budget_conscious': budget_var.get(),
                    'created_via': 'gui',
                    'ai_model_preference': 'gemini-1.5-flash'
                }
                
                # Create user
                user_id = self.user_manager.create_user(
                    name_var.get(),
                    email_var.get(), 
                    area_var.get(),
                    profile
                )
                
                messagebox.showinfo("Success", f"User created successfully!\nUser ID: {user_id}")
                dialog.destroy()
                self.load_users()  # Refresh user list
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to create user: {e}")
        
        ttk.Button(button_frame, text="Create User", command=create_user).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(button_frame, text="Cancel", command=dialog.destroy).pack(side=tk.LEFT)
    
    def process_results(self):
        """Process results from background threads"""
        try:
            while True:
                result_type, result_data = self.result_queue.get_nowait()
                
                if result_type == 'recommendations':
                    self.display_recommendations(result_data)
                elif result_type == 'dashboard':
                    self.display_dashboard(result_data)
                elif result_type == 'error':
                    self.display_error(result_data)
                elif result_type == 'api_usage':
                    self.update_api_usage(result_data)
                    
        except queue.Empty:
            pass
        
        # Schedule next check
        self.root.after(100, self.process_results)
    
    def display_recommendations(self, result):
        """Display recommendations result"""
        self.results_text.delete(1.0, tk.END)
        
        if result['status'] == 'success':
            self.results_text.insert(tk.END, "🎉 AI RECOMMENDATIONS GENERATED SUCCESSFULLY!\n")
            self.results_text.insert(tk.END, "=" * 60 + "\n\n")
            
            # Format and display the result
            if isinstance(result['data'], dict):
                formatted_result = json.dumps(result['data'], indent=2, ensure_ascii=False)
            else:
                formatted_result = str(result['data'])
            
            self.results_text.insert(tk.END, formatted_result)
            self.update_status("✅ Recommendations generated successfully", "success")
            
        elif result['status'] == 'success_fallback':
            self.results_text.insert(tk.END, "⚠️ RECOMMENDATIONS (FALLBACK MODE)\n")
            self.results_text.insert(tk.END, "=" * 60 + "\n\n")
            self.results_text.insert(tk.END, result['data'].get('recommendations', 'No recommendations available'))
            self.update_status("⚠️ Recommendations generated (fallback mode)", "warning")
            
        else:
            self.display_error(result.get('error', 'Unknown error'))
    
    def display_dashboard(self, result):
        """Display dashboard result"""
        self.results_text.delete(1.0, tk.END)
        
        if result['status'] == 'success':
            self.results_text.insert(tk.END, "📊 USER DASHBOARD\n")
            self.results_text.insert(tk.END, "=" * 60 + "\n\n")
            
            formatted_result = json.dumps(result['data'], indent=2, ensure_ascii=False)
            self.results_text.insert(tk.END, formatted_result)
            self.update_status("✅ Dashboard loaded successfully", "success")
        else:
            self.display_error(result.get('error', 'Unknown error'))
    
    def display_error(self, error_msg):
        """Display error message"""
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, "❌ ERROR OCCURRED\n")
        self.results_text.insert(tk.END, "=" * 60 + "\n\n")
        self.results_text.insert(tk.END, f"Error: {error_msg}\n\n")
        self.results_text.insert(tk.END, "Please try again or check your configuration.")
        
        self.update_status("❌ Operation failed", "error")
    
    def update_status(self, message: str, status_type: str = "info"):
        """Update status message"""
        color_map = {
            "success": "Success.TLabel",
            "warning": "Warning.TLabel",
            "error": "Error.TLabel",
            "info": "Info.TLabel"
        }
        
        style = color_map.get(status_type, "Info.TLabel")
        self.status_label.config(text=message, style=style)
    
    def update_api_usage(self, current_requests: int):
        """Update API usage display"""
        self.api_usage_label.config(text=f"API Calls: {current_requests}/12 per minute")
        
        if current_requests >= 10:
            self.api_usage_label.config(style='Warning.TLabel')
        elif current_requests >= 12:
            self.api_usage_label.config(style='Error.TLabel')
        else:
            self.api_usage_label.config(style='Info.TLabel')

# ===========================
# ENHANCED DATABASE SETUP
# ===========================

def setup_database_tables():
    """Setup required database tables if they don't exist"""
    db_manager = DatabaseManager()
    
    tables = {
        'users': """
        CREATE TABLE IF NOT EXISTS users (
            id VARCHAR(36) PRIMARY KEY,
            name VARCHAR(255) NOT NULL,
            email VARCHAR(255) UNIQUE NOT NULL,
            area VARCHAR(255) NOT NULL,
            profile JSON,
            is_active BOOLEAN DEFAULT TRUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
        )
        """,
        
        'user_stats': """
        CREATE TABLE IF NOT EXISTS user_stats (
            user_id VARCHAR(36) PRIMARY KEY,
            total_coins INT DEFAULT 0,
            total_co2_saved_kg DECIMAL(10,3) DEFAULT 0,
            current_streak INT DEFAULT 0,
            recommendations_accepted INT DEFAULT 0,
            recommendations_total INT DEFAULT 0,
            rank_position INT DEFAULT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
        """,
        
        'user_states': """
        CREATE TABLE IF NOT EXISTS user_states (
            id INT AUTO_INCREMENT PRIMARY KEY,
            user_id VARCHAR(36) NOT NULL,
            current_activity VARCHAR(255),
            context JSON,
            is_active BOOLEAN DEFAULT TRUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
        """,
        
        'recommendations': """
        CREATE TABLE IF NOT EXISTS recommendations (
            id INT AUTO_INCREMENT PRIMARY KEY,
            user_id VARCHAR(36) NOT NULL,
            session_id VARCHAR(36),
            category VARCHAR(100),
            title VARCHAR(500),
            description TEXT,
            impact_metrics JSON,
            feasibility_score DECIMAL(3,2),
            user_preference_score DECIMAL(3,2),
            priority_rank INT,
            status ENUM('pending', 'accepted', 'rejected', 'expired') DEFAULT 'pending',
            expires_at TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
        """,
        
        'recommendation_feedback': """
        CREATE TABLE IF NOT EXISTS recommendation_feedback (
            id INT AUTO_INCREMENT PRIMARY KEY,
            recommendation_id INT NOT NULL,
            user_id VARCHAR(36) NOT NULL,
            response ENUM('accepted', 'rejected', 'skipped') NOT NULL,
            feedback_reason JSON,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (recommendation_id) REFERENCES recommendations(id) ON DELETE CASCADE,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
        """,
        
        'sustainability_actions': """
        CREATE TABLE IF NOT EXISTS sustainability_actions (
            id INT AUTO_INCREMENT PRIMARY KEY,
            action_type VARCHAR(100) NOT NULL,
            title VARCHAR(500) NOT NULL,
            description TEXT,
            impact_metrics JSON,
            category VARCHAR(100),
            is_active BOOLEAN DEFAULT TRUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """,
        
        'user_preferences': """
        CREATE TABLE IF NOT EXISTS user_preferences (
            id INT AUTO_INCREMENT PRIMARY KEY,
            user_id VARCHAR(36) NOT NULL,
            category VARCHAR(100),
            preference_type VARCHAR(100),
            preference_value TEXT,
            confidence_score DECIMAL(3,2),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
        """
    }
    
    try:
        for table_name, create_sql in tables.items():
            db_manager.execute_query(create_sql)
            logger.info(f"✅ Table {table_name} ready")
        
        logger.info("✅ All database tables are ready!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Database setup failed: {e}")
        return False

# ===========================
# SAMPLE DATA CREATION
# ===========================

def create_sample_users(db_manager: DatabaseManager, count: int = 3):
    """Create sample users for testing"""
    import uuid
    
    sample_users = [
        {
            'name': 'Priya Sharma',
            'email': 'priya.sharma@example.com',
            'area': 'Andheri East',
            'profile': {
                'sustainability_goal': 'reduce_carbon_footprint',
                'transport_modes': ['metro', 'bus', 'auto'],
                'vehicle_info': None,
                'dietary_preferences': 'vegetarian',
                'energy_usage': {
                    'monthly_bill': 2800,
                    'has_renewable': False,
                    'home_type': '2bhk_apartment'
                },
                'budget_conscious': True,
                'age_group': '25-35'
            }
        },
        {
            'name': 'Rajesh Kumar',
            'email': 'rajesh.kumar@example.com',
            'area': 'Bandra West',
            'profile': {
                'sustainability_goal': 'save_money',
                'transport_modes': ['car', 'metro', 'taxi'],
                'vehicle_info': {
                    'type': 'compact_car_petrol',
                    'fuel_efficiency': 18.2
                },
                'dietary_preferences': 'non_vegetarian',
                'energy_usage': {
                    'monthly_bill': 4200,
                    'has_renewable': False,
                    'home_type': '3bhk_apartment'
                },
                'budget_conscious': False,
                'age_group': '35-45'
            }
        },
        {
            'name': 'Sneha Patel',
            'email': 'sneha.patel@example.com',
            'area': 'Powai',
            'profile': {
                'sustainability_goal': 'live_healthier',
                'transport_modes': ['cycle', 'metro', 'bus', 'walk'],
                'vehicle_info': None,
                'dietary_preferences': 'vegan',
                'energy_usage': {
                    'monthly_bill': 1800,
                    'has_renewable': True,
                    'home_type': '1bhk_apartment'
                },
                'budget_conscious': True,
                'age_group': '20-30'
            }
        }
    ]
    
    created_users = []
    
    for user_data in sample_users[:count]:
        try:
            user_id = str(uuid.uuid4())
            
            # Insert user
            user_query = """
            INSERT INTO users (id, name, email, area, profile, created_at) 
            VALUES (%s, %s, %s, %s, %s, NOW())
            """
            
            profile_json = json.dumps(user_data['profile'])
            db_manager.execute_query(user_query, (
                user_id, user_data['name'], user_data['email'], 
                user_data['area'], profile_json
            ))
            
            # Insert user stats
            stats_query = """
            INSERT INTO user_stats (user_id, total_coins, total_co2_saved_kg, current_streak, 
                                  recommendations_accepted, recommendations_total)
            VALUES (%s, %s, %s, %s, %s, %s)
            """
            
            import random
            db_manager.execute_query(stats_query, (
                user_id,
                random.randint(50, 500),  # total_coins
                round(random.uniform(5.0, 50.0), 2),  # total_co2_saved_kg
                random.randint(0, 15),  # current_streak
                random.randint(5, 25),  # recommendations_accepted
                random.randint(10, 30)  # recommendations_total
            ))
            
            created_users.append({
                'id': user_id,
                'name': user_data['name'],
                'email': user_data['email']
            })
            
            logger.info(f"✅ Created sample user: {user_data['name']}")
            
        except Exception as e:
            logger.warning(f"Could not create user {user_data['name']}: {e}")
    
    return created_users

# ===========================
# MAIN APPLICATION LAUNCHER
# ===========================

def main():
    """Main function to launch TerraWise GUI"""
    
    print("🌱 Starting TerraWise GUI Application")
    print("🤖 With Gemini API Rate Limiting")
    print("=" * 50)
    
    try:
        # Check environment
        if not os.getenv('GOOGLE_API_KEY'):
            print("❌ GOOGLE_API_KEY environment variable not set")
            print("Please set your Google API key:")
            print("export GOOGLE_API_KEY='your-api-key-here'")
            return
        
        # Setup database
        print("🔧 Setting up database...")
        if not setup_database_tables():
            print("❌ Database setup failed")
            return
        
        # Check if users exist, create samples if needed
        db_manager = DatabaseManager()
        existing_users = db_manager.execute_query("SELECT COUNT(*) as count FROM users WHERE is_active = TRUE")
        
        if existing_users and existing_users[0]['count'] == 0:
            print("👥 No users found, creating sample users...")
            created_users = create_sample_users(db_manager, 3)
            print(f"✅ Created {len(created_users)} sample users")
        
        # Launch GUI
        print("🚀 Launching TerraWise GUI...")
        root = tk.Tk()
        
        # Set app icon and styling
        try:
            root.iconname("TerraWise")
            root.wm_attributes('-alpha', 0.98)  # Slight transparency for modern look
        except:
            pass  # Ignore if not supported
        
        app = TerraWiseGUI(root)
        
        print("✅ GUI launched successfully!")
        print("💡 Tips:")
        print("   - Select a user from the left panel")
        print("   - Choose current activity")
        print("   - Generate AI recommendations (rate limited)")
        print("   - View user dashboard")
        print("   - Create new users as needed")
        print(f"   - API Rate Limit: 12 requests per minute")
        
        # Start the GUI event loop
        root.mainloop()
        
    except KeyboardInterrupt:
        print("\n👋 TerraWise GUI closed by user")
    except Exception as e:
        logger.error(f"GUI startup failed: {e}")
        print(f"❌ Failed to start GUI: {e}")
        print("\nTroubleshooting:")
        print("1. Check GOOGLE_API_KEY is set")
        print("2. Ensure MySQL is running with 'terra' database")
        print("3. Install required packages: tkinter, mysql-connector-python")
        print("4. Check your TerraWise module import path")

if __name__ == "__main__":
    main()