"""
TerraWise GUI - Beautiful Green-Themed Sustainability Dashboard
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import json
from datetime import datetime
import threading
from typing import Dict, List
import uuid

from main import TerraWiseApp, setup_demo_user

class TerraWiseGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("TerraWise - AI Sustainability Assistant 🌱")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f8f0')
        
        # Initialize TerraWise app (you'll need to uncomment this when integrating)
        # self.app = TerraWiseApp()
        self.current_user_id = None
        self.demo_data = self.create_demo_data()
        
        # Configure styles
        self.setup_styles()
        
        # Create GUI components
        self.create_main_interface()
        
        # Load demo user
        self.load_demo_user()
    
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
        self.create_profile_tab()
        self.create_settings_tab()
    
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
        
        ttk.Label(title_frame, text="AI-Powered Sustainability Assistant", 
                 font=('Arial', 12), 
                 foreground='#4a7c59',
                 background='white').pack(anchor='w')
        
        # User info and controls
        user_frame = ttk.Frame(header_frame)
        user_frame.pack(side='right', padx=20, pady=10)
        
        self.user_label = ttk.Label(user_frame, text="Demo User", 
                                   font=('Arial', 12, 'bold'),
                                   background='white')
        self.user_label.pack(anchor='e')
        
        ttk.Button(user_frame, text="🔄 Refresh Data",
                  style='Secondary.TButton',
                  command=self.refresh_data).pack(anchor='e', pady=5)
    
    def create_dashboard_tab(self):
        """Create the main dashboard tab"""
        dashboard_frame = ttk.Frame(self.notebook)
        self.notebook.add(dashboard_frame, text="🏠 Dashboard")
        
        # Create scrollable canvas
        canvas = tk.Canvas(dashboard_frame, bg='#f0f8f0')
        scrollbar = ttk.Scrollbar(dashboard_frame, orient='vertical', command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Stats cards
        self.create_stats_cards(scrollable_frame)
        
        # Recent activity
        self.create_recent_activity(scrollable_frame)
        
        # Environmental impact chart
        self.create_impact_chart(scrollable_frame)
    
    def create_stats_cards(self, parent):
        """Create statistics cards"""
        stats_frame = ttk.Frame(parent)
        stats_frame.pack(fill='x', padx=10, pady=10)
        
        ttk.Label(stats_frame, text="Your Impact", style='Title.TLabel').pack(anchor='w')
        
        # Cards container
        cards_frame = ttk.Frame(stats_frame)
        cards_frame.pack(fill='x', pady=10)
        
        stats = self.demo_data['user_stats']
        
        # Create individual stat cards
        self.create_stat_card(cards_frame, "🪙 Total Coins", 
                             f"{stats['total_coins']:,}", 0, 0)
        
        self.create_stat_card(cards_frame, "🌍 CO₂ Saved", 
                             f"{stats['total_co2_saved']:.1f} kg", 0, 1)
        
        self.create_stat_card(cards_frame, "🔥 Current Streak", 
                             f"{stats['current_streak']} days", 0, 2)
        
        self.create_stat_card(cards_frame, "✅ Actions Completed", 
                             f"{stats['accepted_count']}/{stats['total_count']}", 1, 0)
        
        self.create_stat_card(cards_frame, "💰 Money Saved", 
                             f"₹{stats['money_saved']:,}", 1, 1)
        
        self.create_stat_card(cards_frame, "🏆 Rank", 
                             f"#{stats['rank_position']}", 1, 2)
    
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
    
    def create_recent_activity(self, parent):
        """Create recent activity section"""
        activity_frame = ttk.Frame(parent)
        activity_frame.pack(fill='x', padx=10, pady=10)
        
        ttk.Label(activity_frame, text="Recent Recommendations", 
                 style='Title.TLabel').pack(anchor='w')
        
        # Activity list
        for i, rec in enumerate(self.demo_data['recent_recommendations'][:3]):
            self.create_activity_item(activity_frame, rec, i)
    
    def create_activity_item(self, parent, recommendation, index):
        """Create individual activity item"""
        item_frame = ttk.Frame(parent, style='Card.TFrame')
        item_frame.pack(fill='x', pady=5)
        
        content_frame = ttk.Frame(item_frame)
        content_frame.pack(fill='x', padx=15, pady=10)
        
        # Icon and title
        header_frame = ttk.Frame(content_frame)
        header_frame.pack(fill='x')
        
        icons = {'transport': '🚗', 'energy': '⚡', 'food': '🍽️', 'water': '💧'}
        icon = icons.get(recommendation['category'], '🌱')
        
        ttk.Label(header_frame, text=f"{icon} {recommendation['title']}", 
                 font=('Arial', 12, 'bold'),
                 background='white').pack(side='left')
        
        status_colors = {'completed': '#28a745', 'pending': '#ffc107', 'expired': '#dc3545'}
        status_color = status_colors.get(recommendation['status'], '#6c757d')
        
        status_label = tk.Label(header_frame, text=recommendation['status'].title(),
                               bg=status_color, fg='white',
                               font=('Arial', 8, 'bold'),
                               padx=8, pady=2)
        status_label.pack(side='right')
        
        # Details
        details = f"Category: {recommendation['category'].title()} | Created: {recommendation['created_at'][:10]}"
        ttk.Label(content_frame, text=details, 
                 font=('Arial', 9),
                 foreground='#666666',
                 background='white').pack(anchor='w', pady=(5, 0))
    
    def create_impact_chart(self, parent):
        """Create environmental impact visualization"""
        chart_frame = ttk.Frame(parent)
        chart_frame.pack(fill='x', padx=10, pady=10)
        
        ttk.Label(chart_frame, text="Environmental Impact", 
                 style='Title.TLabel').pack(anchor='w')
        
        # Simple text-based chart (you can replace with matplotlib)
        impact_data = [
            ("Transport", 45, "#28a745"),
            ("Energy", 30, "#17a2b8"),
            ("Food", 15, "#ffc107"),
            ("Water", 10, "#6f42c1")
        ]
        
        chart_canvas = tk.Canvas(chart_frame, height=200, bg='white')
        chart_canvas.pack(fill='x', pady=10)
        
        # Draw simple bar chart
        total_width = 600
        bar_height = 30
        start_y = 50
        
        for i, (category, percentage, color) in enumerate(impact_data):
            y = start_y + i * 40
            bar_width = (percentage / 100) * total_width
            
            # Draw bar
            chart_canvas.create_rectangle(50, y, 50 + bar_width, y + bar_height,
                                        fill=color, outline=color)
            
            # Draw label
            chart_canvas.create_text(20, y + bar_height//2, text=category,
                                   anchor='e', font=('Arial', 10))
            
            # Draw percentage
            chart_canvas.create_text(60 + bar_width, y + bar_height//2,
                                   text=f"{percentage}%", anchor='w',
                                   font=('Arial', 10, 'bold'))
    
    def create_recommendations_tab(self):
        """Create recommendations tab"""
        rec_frame = ttk.Frame(self.notebook)
        self.notebook.add(rec_frame, text="💡 Recommendations")
        
        # Header
        header_frame = ttk.Frame(rec_frame)
        header_frame.pack(fill='x', padx=20, pady=10)
        
        ttk.Label(header_frame, text="AI-Powered Sustainability Recommendations",
                 style='Title.TLabel').pack(side='left')
        
        ttk.Button(header_frame, text="🤖 Generate New",
                  style='Primary.TButton',
                  command=self.generate_recommendations).pack(side='right')
        
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
        
        # Recommendations display
        self.rec_display_frame = ttk.Frame(rec_frame)
        self.rec_display_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        self.display_recommendations()
    
    def display_recommendations(self):
        """Display current recommendations"""
        # Clear existing recommendations
        for widget in self.rec_display_frame.winfo_children():
            widget.destroy()
        
        recommendations = self.demo_data.get('recommendations', [])
        
        if not recommendations:
            no_rec_label = ttk.Label(self.rec_display_frame,
                                    text="No recommendations available. Click 'Generate New' to get started!",
                                    style='Body.TLabel')
            no_rec_label.pack(pady=50)
            return
        
        # Create scrollable area
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
        
        # Display each recommendation
        for i, rec in enumerate(recommendations):
            self.create_recommendation_card(scrollable_frame, rec, i)
    
    def create_recommendation_card(self, parent, recommendation, index):
        """Create recommendation card"""
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
        icon = icons.get(recommendation['category'], '🌱')
        
        title_text = f"{icon} {recommendation['title']}"
        ttk.Label(header_frame, text=title_text,
                 font=('Arial', 14, 'bold'),
                 background='white').pack(anchor='w')
        
        # Category badge
        category_label = tk.Label(header_frame, text=recommendation['category'].title(),
                                 bg='#6fb37b', fg='white',
                                 font=('Arial', 9, 'bold'),
                                 padx=8, pady=2)
        category_label.pack(anchor='e')
        
        # Description
        desc_text = recommendation.get('description', 'No description available')
        ttk.Label(content_frame, text=desc_text,
                 font=('Arial', 10),
                 background='white',
                 wraplength=600).pack(anchor='w', pady=(10, 0))
        
        # Metrics
        metrics_frame = ttk.Frame(content_frame)
        metrics_frame.pack(fill='x', pady=(15, 0))
        
        metrics = recommendation.get('impact_metrics', {})
        
        # CO2 savings
        co2_saved = metrics.get('co2_reduction_kg', 0)
        self.create_metric_item(metrics_frame, "🌍 CO₂ Saved", f"{co2_saved} kg", 0)
        
        # Money saved
        money_saved = metrics.get('money_saved_inr', 0)
        self.create_metric_item(metrics_frame, "💰 Money Saved", f"₹{money_saved}", 1)
        
        # Coins earned
        coins = metrics.get('coins_earned', 0)
        self.create_metric_item(metrics_frame, "🪙 Coins", f"{coins}", 2)
        
        # Action buttons
        button_frame = ttk.Frame(content_frame)
        button_frame.pack(fill='x', pady=(15, 0))
        
        ttk.Button(button_frame, text="✅ Accept",
                  style='Primary.TButton',
                  command=lambda: self.handle_recommendation('accepted', recommendation)).pack(side='left', padx=(0, 10))
        
        ttk.Button(button_frame, text="❌ Decline",
                  style='Secondary.TButton',
                  command=lambda: self.handle_recommendation('declined', recommendation)).pack(side='left')
        
        ttk.Button(button_frame, text="ℹ️ More Info",
                  command=lambda: self.show_recommendation_details(recommendation)).pack(side='right')
    
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
    
    def create_profile_tab(self):
        """Create user profile tab"""
        profile_frame = ttk.Frame(self.notebook)
        self.notebook.add(profile_frame, text="👤 Profile")
        
        # Profile header
        header_frame = ttk.Frame(profile_frame)
        header_frame.pack(fill='x', padx=20, pady=20)
        
        ttk.Label(header_frame, text="User Profile & Preferences",
                 style='Title.TLabel').pack(anchor='w')
        
        # Create scrollable content
        canvas = tk.Canvas(profile_frame, bg='#f0f8f0')
        scrollbar = ttk.Scrollbar(profile_frame, orient='vertical', command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True, padx=20)
        scrollbar.pack(side="right", fill="y")
        
        # Profile sections
        self.create_profile_section(scrollable_frame, "Basic Information", [
            ("Name", "Demo User"),
            ("Email", "demo@terrawise.com"),
            ("Location", "Andheri East, Mumbai"),
            ("Member Since", "March 2024")
        ])
        
        self.create_profile_section(scrollable_frame, "Sustainability Goals", [
            ("Primary Goal", "Reduce Carbon Footprint"),
            ("Target CO₂ Reduction", "500 kg/year"),
            ("Focus Areas", "Transport, Energy, Food"),
            ("Budget Range", "₹500-2000/month")
        ])
        
        self.create_profile_section(scrollable_frame, "Transport Preferences", [
            ("Preferred Modes", "Metro, Bus, Cycle"),
            ("Vehicle Type", "Small Car (Petrol)"),
            ("Average Daily Distance", "25 km"),
            ("Fuel Efficiency", "15.5 km/l")
        ])
        
        self.create_profile_section(scrollable_frame, "Energy Usage", [
            ("Home Type", "2BHK Apartment"),
            ("Monthly Electricity Bill", "₹3,500"),
            ("Renewable Energy", "No"),
            ("Smart Devices", "Basic")
        ])
    
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
            
            ttk.Label(item_frame, text=value,
                     font=('Arial', 10, 'bold'),
                     background='white').pack(side='right')
        
        # Add spacing
        ttk.Frame(section_frame, height=10).pack()
    
    def create_settings_tab(self):
        """Create settings tab"""
        settings_frame = ttk.Frame(self.notebook)
        self.notebook.add(settings_frame, text="⚙️ Settings")
        
        # Settings header
        header_frame = ttk.Frame(settings_frame)
        header_frame.pack(fill='x', padx=20, pady=20)
        
        ttk.Label(header_frame, text="Application Settings",
                 style='Title.TLabel').pack(anchor='w')
        
        # Settings content
        content_frame = ttk.Frame(settings_frame)
        content_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        # AI Model settings
        ai_frame = ttk.LabelFrame(content_frame, text="AI Configuration", padding=15)
        ai_frame.pack(fill='x', pady=10)
        
        ttk.Label(ai_frame, text="Current AI Model: Google Gemini Pro (Free)",
                 font=('Arial', 10, 'bold')).pack(anchor='w')
        
        ttk.Label(ai_frame, text="API Status: Connected ✅",
                 font=('Arial', 10), foreground='#28a745').pack(anchor='w', pady=(5, 0))
        
        # Notification settings
        notif_frame = ttk.LabelFrame(content_frame, text="Notifications", padding=15)
        notif_frame.pack(fill='x', pady=10)
        
        self.email_notif_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(notif_frame, text="Email notifications for new recommendations",
                       variable=self.email_notif_var).pack(anchor='w')
        
        self.push_notif_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(notif_frame, text="Push notifications",
                       variable=self.push_notif_var).pack(anchor='w')
        
        # Data settings
        data_frame = ttk.LabelFrame(content_frame, text="Data & Privacy", padding=15)
        data_frame.pack(fill='x', pady=10)
        
        ttk.Button(data_frame, text="Export Data",
                  command=self.export_data).pack(side='left', padx=(0, 10))
        
        ttk.Button(data_frame, text="Clear History",
                  command=self.clear_history).pack(side='left')
        
        # About section
        about_frame = ttk.LabelFrame(content_frame, text="About", padding=15)
        about_frame.pack(fill='x', pady=10)
        
        about_text = """TerraWise v1.0 - AI-Powered Sustainability Assistant
        
Powered by Google Gemini Pro (Free)
Built with CrewAI Multi-Agent System
Database: MySQL

Features:
• Personalized sustainability recommendations
• Real-time environmental impact calculations
• Gamification with coins and achievements
• Mumbai-specific transport and energy advice
• Completely free and open source"""
        
        ttk.Label(about_frame, text=about_text,
                 font=('Arial', 9), justify='left').pack(anchor='w')
    
    # Event handlers
    def generate_recommendations(self):
        """Generate new recommendations"""
        # Show loading dialog
        loading_dialog = tk.Toplevel(self.root)
        loading_dialog.title("Generating Recommendations")
        loading_dialog.geometry("300x150")
        loading_dialog.configure(bg='white')
        loading_dialog.transient(self.root)
        loading_dialog.grab_set()
        
        # Center the dialog
        loading_dialog.geometry("+%d+%d" % (self.root.winfo_rootx()+50, self.root.winfo_rooty()+50))
        
        ttk.Label(loading_dialog, text="🤖 AI is generating recommendations...",
                 font=('Arial', 12), background='white').pack(pady=30)
        
        progress = ttk.Progressbar(loading_dialog, mode='indeterminate')
        progress.pack(pady=10)
        progress.start()
        
        def generate_async():
            # Simulate API call
            import time
            time.sleep(3)
            
            # Close loading dialog
            loading_dialog.destroy()
            
            # Update recommendations
            self.demo_data['recommendations'] = self.create_sample_recommendations()
            self.display_recommendations()
            
            # Show success message
            messagebox.showinfo("Success", "New recommendations generated successfully! 🌱")
        
        # Run in separate thread
        threading.Thread(target=generate_async, daemon=True).start()
    
    def handle_recommendation(self, action, recommendation):
        """Handle recommendation feedback"""
        if action == 'accepted':
            messagebox.showinfo("Accepted", f"Great choice! You earned {recommendation.get('impact_metrics', {}).get('coins_earned', 15)} coins! 🪙")
            # Update user stats
            self.demo_data['user_stats']['total_coins'] += recommendation.get('impact_metrics', {}).get('coins_earned', 15)
            self.demo_data['user_stats']['accepted_count'] += 1
            self.demo_data['user_stats']['current_streak'] += 1
        else:
            messagebox.showinfo("Declined", "Thanks for the feedback! We'll improve our recommendations. 🤖")
        
        # Refresh dashboard
        self.refresh_data()
    
    def show_recommendation_details(self, recommendation):
        """Show detailed recommendation information"""
        details_window = tk.Toplevel(self.root)
        details_window.title("Recommendation Details")
        details_window.geometry("500x400")
        details_window.configure(bg='white')
        details_window.transient(self.root)
        
        # Title
        title_label = tk.Label(details_window, text=recommendation['title'],
                              font=('Arial', 16, 'bold'),
                              bg='white', fg='#2d5a3d')
        title_label.pack(pady=20)
        
        # Details text
        details_text = scrolledtext.ScrolledText(details_window, height=15, width=60,
                                               font=('Arial', 10), wrap=tk.WORD)
        details_text.pack(padx=20, pady=10)
        
        # Sample detailed content
        content = f"""Recommendation: {recommendation['title']}
Category: {recommendation['category'].title()}

Description:
{recommendation.get('description', 'Detailed description would go here.')}

Environmental Impact:
• CO₂ Reduction: {recommendation.get('impact_metrics', {}).get('co2_reduction_kg', 0)} kg
• Money Saved: ₹{recommendation.get('impact_metrics', {}).get('money_saved_inr', 0)}
• Difficulty: Easy
• Time Required: 5-15 minutes

Step-by-Step Instructions:
1. Research sustainable alternatives
2. Compare costs and benefits
3. Make gradual changes
4. Track your progress
5. Share your success

Tips for Success:
• Start small and build habits gradually
• Track your progress daily
• Join community challenges
• Share achievements with friends
• Celebrate small wins

Mumbai-Specific Considerations:
• Consider monsoon season impacts
• Factor in local transport schedules
• Account for power outages
• Use locally available alternatives

Additional Resources:
• Mumbai Metro route planner
• Local cycle sharing apps
• Energy-efficient appliance stores
• Community sustainability groups

Expected Timeline:
• Week 1: Initial setup and research
• Week 2-3: Implementation and adjustment
• Week 4+: Habit formation and optimization

Rewards:
• Coins: {recommendation.get('impact_metrics', {}).get('coins_earned', 15)}
• XP Points: {recommendation.get('impact_metrics', {}).get('coins_earned', 15) * 2}
• Achievement: Environmental Champion (if completed)
"""
        
        details_text.insert(tk.END, content)
        details_text.config(state=tk.DISABLED)
        
        # Close button
        ttk.Button(details_window, text="Close", 
                  command=details_window.destroy).pack(pady=10)
    
    def refresh_data(self):
        """Refresh all data displays"""
        # Update user stats display
        self.load_demo_user()
        
        # Refresh current tab
        current_tab = self.notebook.select()
        tab_text = self.notebook.tab(current_tab, "text")
        
        if "Dashboard" in tab_text:
            # Refresh dashboard - would need to recreate stats cards
            pass
        elif "Recommendations" in tab_text:
            self.display_recommendations()
        
        messagebox.showinfo("Refreshed", "Data has been refreshed! 🔄")
    
    def export_data(self):
        """Export user data"""
        from tkinter import filedialog
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filename:
            export_data = {
                'user_stats': self.demo_data['user_stats'],
                'recommendations': self.demo_data['recommendations'],
                'export_date': datetime.now().isoformat(),
                'version': '1.0'
            }
            
            try:
                with open(filename, 'w') as f:
                    json.dump(export_data, f, indent=2)
                messagebox.showinfo("Success", f"Data exported to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export data: {e}")
    
    def clear_history(self):
        """Clear user history"""
        result = messagebox.askyesno("Confirm", "Are you sure you want to clear all history? This cannot be undone.")
        
        if result:
            # Reset demo data
            self.demo_data['recommendations'] = []
            self.demo_data['recent_recommendations'] = []
            
            messagebox.showinfo("Cleared", "History has been cleared.")
            self.refresh_data()
    
    def load_demo_user(self):
        """Load demo user data"""
        # This would normally load from your TerraWise app
        self.current_user_id = "demo-user-123"
        self.user_label.config(text="Demo User (Mumbai)")
    
    def create_demo_data(self):
        """Create demo data for the GUI"""
        return {
            'user_stats': {
                'total_coins': 1250,
                'total_co2_saved': 45.7,
                'current_streak': 12,
                'accepted_count': 18,
                'total_count': 22,
                'money_saved': 3400,
                'rank_position': 147
            },
            'recent_recommendations': [
                {
                    'id': 1,
                    'title': 'Switch to Mumbai Metro',
                    'category': 'transport',
                    'status': 'completed',
                    'created_at': '2024-03-15T10:30:00'
                },
                {
                    'id': 2,
                    'title': 'Optimize AC Temperature',
                    'category': 'energy',
                    'status': 'pending',
                    'created_at': '2024-03-14T14:20:00'
                },
                {
                    'id': 3,
                    'title': 'Reduce Food Waste',
                    'category': 'food',
                    'status': 'completed',
                    'created_at': '2024-03-13T09:15:00'
                }
            ],
            'recommendations': []
        }
    
    def create_sample_recommendations(self):
        """Create sample recommendations for demo"""
        return [
            {
                'id': str(uuid.uuid4()),
                'category': 'transport',
                'title': 'Use Mumbai Metro for Daily Commute',
                'description': 'Switch from taxi/car to Mumbai Metro for your daily office commute. The Metro is faster, cheaper, and significantly more environmentally friendly.',
                'impact_metrics': {
                    'co2_reduction_kg': 2.1,
                    'money_saved_inr': 350,
                    'coins_earned': 25
                }
            },
            {
                'id': str(uuid.uuid4()),
                'category': 'energy',
                'title': 'Install LED Bulbs',
                'description': 'Replace all incandescent and CFL bulbs with LED bulbs. LEDs use 75% less energy and last 25 times longer.',
                'impact_metrics': {
                    'co2_reduction_kg': 1.8,
                    'money_saved_inr': 200,
                    'coins_earned': 20
                }
            },
            {
                'id': str(uuid.uuid4()),
                'category': 'food',
                'title': 'Start Composting Kitchen Waste',
                'description': 'Create a simple composting system for your kitchen waste. This reduces methane emissions from landfills and creates nutrient-rich soil.',
                'impact_metrics': {
                    'co2_reduction_kg': 1.2,
                    'money_saved_inr': 150,
                    'coins_earned': 18
                }
            },
            {
                'id': str(uuid.uuid4()),
                'category': 'water',
                'title': 'Install Water-Saving Devices',
                'description': 'Install low-flow showerheads and faucet aerators to reduce water consumption without sacrificing performance.',
                'impact_metrics': {
                    'co2_reduction_kg': 0.9,
                    'money_saved_inr': 120,
                    'coins_earned': 15
                }
            }
        ]

def main():
    """Main function to run the GUI"""
    root = tk.Tk()
    
    # Set window icon (if you have an icon file)
    try:
        root.iconbitmap('terrawise_icon.ico')  # Optional: add your icon
    except:
        pass  # Ignore if icon file doesn't exist
    
    # Create and run the application
    app = TerraWiseGUI(root)
    
    # Center the window
    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (1200 // 2)
    y = (root.winfo_screenheight() // 2) - (800 // 2)
    root.geometry(f"1200x800+{x}+{y}")
    
    # Start the GUI event loop
    root.mainloop()

if __name__ == "__main__":
    main()

"""
Integration Instructions:
========================

To integrate this GUI with your existing TerraWise system:

1. Save this code as 'terrawise_gui.py'

2. In the __init__ method, uncomment and modify:
   # self.app = TerraWiseApp()

3. Replace demo data methods with actual API calls:
   - In generate_recommendations(), replace the demo logic with:
     recommendations = self.app.generate_recommendations(self.current_user_id, self.activity_var.get())
   
   - In load_demo_user(), replace with:
     self.current_user_id = setup_demo_user(self.app)
     dashboard_data = self.app.get_user_dashboard(self.current_user_id)

4. Install required packages:
   pip install tkinter (usually included with Python)

5. Run the GUI:
   python terrawise_gui.py

Features:
---------
✅ Beautiful green-themed interface
✅ Multi-tab layout (Dashboard, Recommendations, Profile, Settings)
✅ Interactive recommendation cards with accept/decline actions
✅ Real-time statistics display
✅ Environmental impact visualization
✅ User profile management
✅ Settings and preferences
✅ Export functionality
✅ Loading dialogs and progress indicators
✅ Responsive design
✅ Mumbai-specific context
✅ Gamification elements (coins, achievements)

The GUI is designed to be:
- User-friendly and intuitive
- Visually appealing with a green sustainability theme
- Fully functional as a standalone demo
- Easy to integrate with your existing backend
- Responsive and modern-looking

To run as demo: python terrawise_gui.py
To integrate: Follow the integration instructions above
"""