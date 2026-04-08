import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw, ImageFilter, ImageEnhance
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import json
import os
import random
import math

# ==============================
# CONFIGURATION
# ==============================
MODEL_PATH = "models/leaf_cnn_model2_improved.h5"
CLASS_INDICES_PATH = "models/class_indices.json"
IMG_SIZE = (224, 224)

# Modern nature-inspired color palette
COLORS = {
    'primary_dark': '#1b5e20',
    'primary': '#2e7d32',
    'primary_light': '#4caf50',
    'accent': '#66bb6a',
    'accent_light': '#81c784',
    'bg_start': '#e8f5e9',
    'bg_end': '#f1f8e9',
    'white': '#ffffff',
    'text_dark': '#1b5e20',
    'text_light': '#558b2f',
    'shadow': '#c8e6c9',
    'yellow': '#ffd54f',
    'orange': '#ff9800',
    'red': '#f44336',
    'card_bg': '#fafafa',
    'excellent': '#4caf50',
    'good': '#8bc34a',
    'fair': '#ffc107',
    'poor': '#ff9800',
    'very_poor': '#f44336'
}

# Load model and class mapping
try:
    model = load_model(MODEL_PATH)
    print(f"✓ Model loaded from: {MODEL_PATH}")
    
    if os.path.exists(CLASS_INDICES_PATH):
        with open(CLASS_INDICES_PATH, 'r') as f:
            class_mapping = json.load(f)
            class_labels = list(class_mapping.keys())
        print(f"✓ Class mapping loaded: {class_labels}")
    else:
        class_labels = ['entire', 'fascicle', 'lobed', 'palmate', 'pinnate', 'trifoliate']
        print("⚠ Using default class labels")
    
    leaf_descriptions = {
        'entire': '🍃 Entire: Simple leaves with smooth, uninterrupted edges. Found in magnolias, dogwoods, and many tropical plants.',
        'fascicle': '🌲 Fascicle: Needle-like leaves growing in tight clusters. Characteristic of pine trees and conifers.',
        'lobed': '🍁 Lobed: Leaves with deep indentations creating distinct sections. Iconic in oak and maple trees.',
        'palmate': '🖐️ Palmate: Lobes or leaflets radiating from a central point like fingers on a hand. Seen in maples and horse chestnuts.',
        'pinnate': '🌾 Pinnate: Compound leaves with leaflets arranged along both sides of a central stem. Common in roses and ash trees.',
        'trifoliate': '☘️ Trifoliate: Leaves with exactly three leaflets. Found in clovers, strawberries, and poison ivy.'
    }
    
except Exception as e:
    model = None
    class_labels = []
    leaf_descriptions = {}
    print(f"❌ Error loading model: {e}")

# ==============================
# FLOATING LEAF PARTICLE
# ==============================
class FloatingLeaf:
    """Animated background leaf particle"""
    def __init__(self, canvas):
        self.canvas = canvas
        self.reset_position()
        
    def reset_position(self):
        """Initialize or reset particle position"""
        width = self.canvas.winfo_width() or 1200
        height = self.canvas.winfo_height() or 800
        
        self.x = random.randint(0, width)
        self.y = random.randint(-height, 0)
        self.size = random.randint(12, 25)
        self.speed = random.uniform(0.5, 1.5)
        self.swing = random.uniform(0.3, 1.0)
        self.offset = random.uniform(0, math.pi * 2)
        self.rotation = random.randint(0, 360)
        self.rotation_speed = random.uniform(-1.5, 1.5)
        self.color = random.choice(['#81c784', '#a5d6a7', '#c8e6c9', '#66bb6a'])
        self.opacity = random.randint(30, 80)
        
        self.create_leaf()
        
    def create_leaf(self):
        """Create leaf shape"""
        img = Image.new('RGBA', (self.size * 2, self.size * 2), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        points = [
            (self.size, 0),
            (self.size * 1.5, self.size * 0.5),
            (self.size * 1.7, self.size),
            (self.size * 1.5, self.size * 1.5),
            (self.size, self.size * 2),
            (self.size * 0.5, self.size * 1.5),
            (self.size * 0.3, self.size),
            (self.size * 0.5, self.size * 0.5)
        ]
        
        rgb = tuple(int(self.color[i:i+2], 16) for i in (1, 3, 5))
        draw.polygon(points, fill=rgb + (self.opacity,))
        
        rotated = img.rotate(self.rotation, expand=True)
        self.photo = ImageTk.PhotoImage(rotated)
        self.img_id = self.canvas.create_image(self.x, self.y, image=self.photo)
        
    def update(self):
        """Update particle position"""
        height = self.canvas.winfo_height() or 800
        
        self.y += self.speed
        self.x += math.sin(self.y * 0.01 + self.offset) * self.swing
        self.rotation += self.rotation_speed
        
        if self.y > height + 50:
            self.canvas.delete(self.img_id)
            self.reset_position()
            return
        
        self.canvas.coords(self.img_id, self.x, self.y)

# ==============================
# GRADIENT BUTTON CLASS
# ==============================
class GradientButton(tk.Canvas):
    """Modern gradient button with hover effects"""
    def __init__(self, parent, text, command, **kwargs):
        super().__init__(parent, height=50, highlightthickness=0, **kwargs)
        self.command = command
        self.text = text
        self.is_hovered = False
        
        self.create_gradient()
        self.text_id = self.create_text(
            0, 25, text=text,
            font=("Arial", 12, "bold"),
            fill='white'
        )
        
        self.bind('<Button-1>', lambda e: command())
        self.bind('<Enter>', self.on_hover)
        self.bind('<Leave>', self.on_leave)
        self.bind('<Configure>', self.on_resize)
        
    def create_gradient(self):
        """Create gradient background"""
        width = self.winfo_width() or 200
        for i in range(50):
            ratio = i / 50
            r = int(46 + (76 - 46) * ratio)
            g = int(125 + (156 - 125) * ratio)
            b = int(50 + (66 - 50) * ratio)
            color = f'#{r:02x}{g:02x}{b:02x}'
            self.create_line(0, i, width, i, fill=color)
        
    def on_resize(self, event):
        """Handle button resize"""
        self.delete('all')
        self.create_gradient()
        self.text_id = self.create_text(
            event.width // 2, 25, text=self.text,
            font=("Arial", 12, "bold"),
            fill='white'
        )
        
    def on_hover(self, event):
        """Hover effect"""
        self.config(cursor='hand2')
        width = self.winfo_width()
        for i in range(50):
            ratio = i / 50
            r = int(76 + (102 - 76) * ratio)
            g = int(175 + (187 - 175) * ratio)
            b = int(80 + (106 - 80) * ratio)
            color = f'#{r:02x}{g:02x}{b:02x}'
            self.create_line(0, i, width, i, fill=color)
        self.tag_raise(self.text_id)
        
    def on_leave(self, event):
        """Leave hover"""
        self.config(cursor='')
        self.delete('all')
        self.create_gradient()
        width = self.winfo_width()
        self.text_id = self.create_text(
            width // 2, 25, text=self.text,
            font=("Arial", 12, "bold"),
            fill='white'
        )

# ==============================
# MAIN GUI CLASS - FIXED RESPONSIVENESS
# ==============================
class EnhancedLeafGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🌿 Advanced Leaf Recognition System - AI Powered")
        self.root.geometry("1250x700")  # Better default size
        root.resizable(False, False)  # prevent resizing
        
        # Variables
        self.current_image = None
        self.current_photo = None
        self.floating_leaves = []
        self.top3_bars = []
        self.animation_running = True
        self.prediction_history = []
        
        # Configure root grid weights for full responsiveness
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        
        # Bind resize event
        self.root.bind('<Configure>', self.on_window_resize)
        
        # Create gradient background canvas
        self.bg_canvas = tk.Canvas(
            self.root,
            bg=COLORS['bg_start'],
            highlightthickness=0
        )
        self.bg_canvas.grid(row=0, column=0, sticky='nsew')
        
        self.draw_gradient_background()
        
        # Initialize floating leaves
        self.root.after(100, self.init_floating_leaves)
        
        # Create scrollable main container
        self.create_scrollable_container()
        
        # Create main layout
        self.create_main_layout()
        
        # Start animations
        self.animate()
        
    def draw_gradient_background(self):
        """Draw smooth gradient background"""
        self.bg_canvas.delete('gradient')
        width = self.bg_canvas.winfo_width() or 1200
        height = self.bg_canvas.winfo_height() or 800
        
        for i in range(0, height, 2):  # Draw every 2 pixels for performance
            ratio = i / height
            r = int(232 + (241 - 232) * ratio)
            g = int(245 + (248 - 245) * ratio)
            b = int(233 + (233 - 233) * ratio)
            color = f'#{r:02x}{g:02x}{b:02x}'
            self.bg_canvas.create_line(0, i, width, i, fill=color, width=2, tags='gradient')
        
        self.bg_canvas.tag_lower('gradient')
    
    def init_floating_leaves(self):
        """Initialize floating leaf particles"""
        for _ in range(12):  # Reduced for better performance
            leaf = FloatingLeaf(self.bg_canvas)
            self.floating_leaves.append(leaf)
    
    def create_scrollable_container(self):
        """Create a scrollable container for content"""
        # Main frame that will contain everything
        self.main_frame = tk.Frame(self.bg_canvas, bg=COLORS['bg_start'])
        
        # Use place for better control, but with proper positioning
        self.bg_canvas.create_window(
            0, 0,
            window=self.main_frame,
            anchor='nw',
            tags='main_window'
        )
        
        # Bind to update scroll region
        self.main_frame.bind('<Configure>', self.on_frame_configure)
        
    def on_frame_configure(self, event=None):
        """Update canvas scroll region when frame size changes"""
        self.bg_canvas.configure(scrollregion=self.bg_canvas.bbox('all'))
        
        # Center the content when window is larger than content
        canvas_width = self.bg_canvas.winfo_width()
        frame_width = self.main_frame.winfo_width()
        
        if canvas_width > frame_width:
            x_pos = max((canvas_width - frame_width) // 2 - 30, -frame_width)  # shift 50 px left
            self.bg_canvas.coords('main_window', x_pos, 0)
        else:
            x_pos = 0
            
        self.bg_canvas.coords('main_window', x_pos, 0)
    
    def create_main_layout(self):
        """Create responsive main layout"""
        # Configure main_frame grid weights
        self.main_frame.grid_columnconfigure(0, weight=1)
        
        self.create_header()
        self.create_content_area()
        self.create_footer()
    
    def create_header(self):
        """Create header"""
        header = tk.Frame(self.main_frame, bg=COLORS['bg_start'])
        header.grid(row=0, column=0, sticky='ew', padx=20, pady=(20, 10))
        
        title_frame = tk.Frame(header, bg=COLORS['bg_start'])
        title_frame.pack()
        
        # Shadow
        shadow = tk.Label(
            title_frame,
            text="🌿 LEAF RECOGNITION SYSTEM",
            font=("Arial", 32, "bold"),
            bg=COLORS['bg_start'],
            fg=COLORS['shadow']
        )
        shadow.place(x=3, y=3)
        
        # Main title
        self.title_label = tk.Label(
            title_frame,
            text="🌿 LEAF RECOGNITION SYSTEM",
            font=("Arial", 32, "bold"),
            bg=COLORS['bg_start'],
            fg=COLORS['primary']
        )
        self.title_label.pack()
        
        # Subtitle
        subtitle = tk.Label(
            header,
            text="Transfer Learning • MobileNetV2 • Deep Neural Network",
            font=("Arial", 11, "italic"),
            bg=COLORS['bg_start'],
            fg=COLORS['text_light']
        )
        subtitle.pack(pady=(5, 0))
        
        # Decorative line
        line = tk.Canvas(header, height=3, bg=COLORS['bg_start'], highlightthickness=0)
        line.pack(fill='x', padx=150, pady=10)
        line.create_rectangle(0, 0, 1000, 3, fill=COLORS['accent'], outline='')
    
    def create_content_area(self):
        """Create content area with proper grid weights"""
        content = tk.Frame(self.main_frame, bg=COLORS['bg_start'])
        content.grid(row=1, column=0, sticky='nsew', padx=20, pady=10)
        
        # CRITICAL: Configure grid weights properly
        content.grid_rowconfigure(0, weight=1)
        content.grid_columnconfigure(0, weight=1)
        content.grid_columnconfigure(1, weight=1)
        content.grid_columnconfigure(2, weight=0)
        
        # Configure main_frame to expand
        self.main_frame.grid_rowconfigure(1, weight=1)
        
        self.create_image_panel(content)
        self.create_prediction_panel(content)
        self.create_stats_panel(content)
    
    def create_image_panel(self, parent):
        """Create image upload panel"""
        left_panel = tk.Frame(parent, bg=COLORS['white'], relief='raised', bd=2)
        left_panel.grid(row=0, column=0, sticky='nsew', padx=(0, 5))
        
        # Configure internal grid
        left_panel.grid_rowconfigure(1, weight=1)
        left_panel.grid_columnconfigure(0, weight=1)
        
        panel_title = tk.Label(
            left_panel,
            text="📸 IMAGE UPLOAD",
            font=("Arial", 12, "bold"),
            bg=COLORS['bg_start'],
            fg=COLORS['primary'],
            pady=8
        )
        panel_title.grid(row=0, column=0, sticky='ew')
        
        # Image display
        self.image_frame = tk.Frame(left_panel, bg=COLORS['card_bg'])
        self.image_frame.grid(row=1, column=0, sticky='nsew', padx=10, pady=10)
        
        # Configure image frame grid
        self.image_frame.grid_rowconfigure(0, weight=1)
        self.image_frame.grid_columnconfigure(0, weight=1)
        
        self.image_label = tk.Label(
            self.image_frame,
            text="🖼️\n\nNo Image Selected\n\nClick below to upload\n\nSupports: JPG, PNG, BMP",
            font=("Arial", 11),
            bg=COLORS['card_bg'],
            fg=COLORS['text_light'],
            relief='solid',
            bd=2
        )
        self.image_label.grid(row=0, column=0, sticky='nsew', padx=5, pady=5)
        
        # Buttons
        btn_container = tk.Frame(left_panel, bg=COLORS['white'])
        btn_container.grid(row=2, column=0, sticky='ew', padx=10, pady=10)
        
        self.upload_btn = GradientButton(
            btn_container,
            "📁  UPLOAD & ANALYZE",
            self.upload_image,
            bg=COLORS['primary']
        )
        self.upload_btn.pack(fill='x', pady=(0, 7))
        
        self.clear_btn = GradientButton(
            btn_container,
            "🗑️  CLEAR ALL",
            self.clear_image,
            bg=COLORS['text_light']
        )
        self.clear_btn.pack(fill='x')
        
        # Image info
        self.image_info = tk.Label(
            left_panel,
            text="📏 Image Size: --- | 📊 Format: ---",
            font=("Arial", 8),
            bg=COLORS['white'],
            fg=COLORS['text_light']
        )
        self.image_info.grid(row=3, column=0, pady=(0, 8))
    
    def create_prediction_panel(self, parent):
        """Create prediction results panel"""
        mid_panel = tk.Frame(parent, bg=COLORS['white'], relief='raised', bd=2)
        mid_panel.grid(row=0, column=1, sticky='nsew', padx=5)
        
        # Configure internal grid
        mid_panel.grid_rowconfigure(3, weight=1)
        mid_panel.grid_columnconfigure(0, weight=1)
        
        panel_title = tk.Label(
            mid_panel,
            text="🔬 CLASSIFICATION RESULTS",
            font=("Arial", 12, "bold"),
            bg=COLORS['bg_start'],
            fg=COLORS['primary'],
            pady=8
        )
        panel_title.grid(row=0, column=0, sticky='ew')
        
        # Main prediction card
        pred_card = tk.Frame(mid_panel, bg='#f1f8e9', relief='solid', bd=2)
        pred_card.grid(row=1, column=0, sticky='ew', padx=10, pady=10)
        
        self.main_pred_label = tk.Label(
            pred_card,
            text="Awaiting image upload...",
            font=("Arial", 18, "bold"),
            bg='#f1f8e9',
            fg=COLORS['text_light'],
            pady=15,
            wraplength=400
        )
        self.main_pred_label.pack()
        
        # Confidence section
        conf_container = tk.Frame(mid_panel, bg=COLORS['white'])
        conf_container.grid(row=2, column=0, sticky='ew', padx=10, pady=5)
        
        conf_header = tk.Frame(conf_container, bg=COLORS['white'])
        conf_header.pack(fill='x')
        
        tk.Label(
            conf_header,
            text="Confidence Level:",
            font=("Arial", 10, "bold"),
            bg=COLORS['white'],
            fg=COLORS['text_dark']
        ).pack(side='left')
        
        self.conf_label = tk.Label(
            conf_header,
            text="0%",
            font=("Arial", 10, "bold"),
            bg=COLORS['white'],
            fg=COLORS['accent']
        )
        self.conf_label.pack(side='right')
        
        self.conf_canvas = tk.Canvas(
            conf_container,
            height=35,
            bg='#e8f5e9',
            highlightthickness=0,
            relief='solid',
            bd=2
        )
        self.conf_canvas.pack(fill='x', pady=5)
        
        self.conf_bar = self.conf_canvas.create_rectangle(
            0, 0, 0, 35,
            fill=COLORS['accent'],
            outline=''
        )
        
        self.conf_text = self.conf_canvas.create_text(
            200, 17,
            text="",
            font=("Arial", 10, "bold"),
            fill='white'
        )
        
        # Confidence rating
        self.conf_rating = tk.Label(
            conf_container,
            text="",
            font=("Arial", 9, "italic"),
            bg=COLORS['white'],
            fg=COLORS['text_light']
        )
        self.conf_rating.pack(anchor='e', pady=(2, 0))
        
        # Top 3 predictions
        top3_frame = tk.Frame(mid_panel, bg=COLORS['white'])
        top3_frame.grid(row=3, column=0, sticky='nsew', padx=10, pady=5)
        
        tk.Label(
            top3_frame,
            text="📊 Top Predictions:",
            font=("Arial", 10, "bold"),
            bg=COLORS['white'],
            fg=COLORS['text_dark']
        ).pack(anchor='w', pady=(0, 7))
        
        self.top3_bars = []
        medals = ['🥇', '🥈', '🥉']
        colors = [COLORS['excellent'], COLORS['good'], COLORS['fair']]
        
        for i in range(3):
            bar_frame = tk.Frame(top3_frame, bg='#f5f5f5', relief='solid', bd=1)
            bar_frame.pack(fill='x', pady=3)
            
            label = tk.Label(
                bar_frame,
                text=f"{medals[i]} ---",
                font=("Arial", 9),
                bg='#f5f5f5',
                fg=COLORS['text_dark'],
                anchor='w',
                padx=8,
                pady=6
            )
            label.pack(fill='x')
            
            canvas = tk.Canvas(bar_frame, height=5, bg='#e0e0e0', highlightthickness=0)
            canvas.pack(fill='x')
            
            bar = canvas.create_rectangle(0, 0, 0, 5, fill=colors[i], outline='')
            
            self.top3_bars.append({'label': label, 'canvas': canvas, 'bar': bar, 'color': colors[i]})
        
        # Description
        desc_frame = tk.Frame(mid_panel, bg='#fffde7', relief='solid', bd=2)
        desc_frame.grid(row=4, column=0, sticky='ew', padx=10, pady=(5, 10))
        
        self.desc_label = tk.Label(
            desc_frame,
            text="💡 Upload a leaf image to see detailed classification and characteristics",
            font=("Arial", 9),
            bg='#fffde7',
            fg=COLORS['text_dark'],
            wraplength=400,
            justify='left',
            padx=10,
            pady=8
        )
        self.desc_label.pack(fill='x')
    
    def create_stats_panel(self, parent):
        """Create statistics panel with fixed width"""
        right_panel = tk.Frame(parent, bg=COLORS['white'], relief='raised', bd=2)
        right_panel.grid(row=0, column=2, sticky='ns', padx=(5, 0))
        
        # Force fixed width
        right_panel.grid_propagate(False)
        right_panel.configure(width=250)
        
        panel_title = tk.Label(
            right_panel,
            text="📈 SESSION STATS",
            font=("Arial", 11, "bold"),
            bg=COLORS['bg_start'],
            fg=COLORS['primary'],
            pady=8
        )
        panel_title.pack(fill='x')
        
        # Stats cards
        stats_container = tk.Frame(right_panel, bg=COLORS['white'])
        stats_container.pack(fill='both', expand=True, padx=8, pady=8)
        
        # Total predictions
        self.create_stat_card(
            stats_container,
            "🎯 Total Predictions",
            "0",
            COLORS['primary']
        )
        
        # Average confidence
        self.create_stat_card(
            stats_container,
            "📊 Avg Confidence",
            "0%",
            COLORS['accent']
        )
        
        # Model info card
        model_card = tk.Frame(stats_container, bg='#e3f2fd', relief='solid', bd=1)
        model_card.pack(fill='x', pady=6)
        
        tk.Label(
            model_card,
            text="🤖 Model Info",
            font=("Arial", 9, "bold"),
            bg='#e3f2fd',
            fg=COLORS['text_dark']
        ).pack(pady=(6, 4))
        
        tk.Label(
            model_card,
            text=f"Classes: {len(class_labels)}\nImage Size: {IMG_SIZE[0]}x{IMG_SIZE[1]}",
            font=("Arial", 8),
            bg='#e3f2fd',
            fg=COLORS['text_dark'],
            justify='center'
        ).pack(pady=(0, 6))
        
        # Tips section
        tips_card = tk.Frame(stats_container, bg='#fff3e0', relief='solid', bd=1)
        tips_card.pack(fill='both', expand=True, pady=6)
        
        tk.Label(
            tips_card,
            text="💡 Tips",
            font=("Arial", 9, "bold"),
            bg='#fff3e0',
            fg=COLORS['text_dark']
        ).pack(pady=(6, 4))
        
        tips_text = (
            "• Use clear, well-lit images\n"
            "• Center the leaf in frame\n"
            "• Avoid blurry photos\n"
            "• Include full leaf structure\n"
            "• Remove background clutter"
        )
        
        tk.Label(
            tips_card,
            text=tips_text,
            font=("Arial", 7),
            bg='#fff3e0',
            fg=COLORS['text_dark'],
            justify='left',
            anchor='w',
            padx=8
        ).pack(pady=(0, 6), fill='x')
        
    def create_stat_card(self, parent, title, value, color):
        """Create a stat card"""
        card = tk.Frame(parent, bg=color, relief='solid', bd=1)
        card.pack(fill='x', pady=4)
        
        tk.Label(
            card,
            text=title,
            font=("Arial", 8),
            bg=color,
            fg='white'
        ).pack(pady=(6, 2))
        
        label = tk.Label(
            card,
            text=value,
            font=("Arial", 16, "bold"),
            bg=color,
            fg='white'
        )
        label.pack(pady=(0, 6))
        
        if "Total" in title:
            self.total_pred_label = label
        elif "Avg" in title:
            self.avg_conf_label = label
    
    def create_footer(self):
        """Create footer"""
        footer = tk.Frame(self.main_frame, bg=COLORS['bg_start'])
        footer.grid(row=2, column=0, sticky='ew', padx=20, pady=(10, 15))
        
        tk.Label(
            footer,
            text="Powered by TensorFlow & Keras  •  Transfer Learning Architecture  •  CSC583 AI Project",
            font=("Arial", 8),
            bg=COLORS['bg_start'],
            fg=COLORS['text_light']
        ).pack()
    
    def on_window_resize(self, event):
        """Handle window resize"""
        if event.widget == self.root:
            self.draw_gradient_background()
            # Update canvas window position
            self.on_frame_configure()
    
    def animate(self):
        """Animation loop"""
        if not self.animation_running:
            return
        
        for leaf in self.floating_leaves:
            leaf.update()
        
        self.root.after(50, self.animate)
    
    def upload_image(self):
        """Handle image upload and prediction"""
        if model is None:
            messagebox.showerror("Error", "Model not loaded! Check MODEL_PATH.")
            return
        
        file_path = filedialog.askopenfilename(
            title="Select Leaf Image",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")]
        )
        
        if not file_path:
            return
        
        try:
            self.main_pred_label.config(text="🔄 Analyzing image...", fg=COLORS['accent'])
            self.desc_label.config(text="⏳ Processing with deep neural network...")
            self.root.update()
            
            img = Image.open(file_path)
            self.current_image = img.copy()
            
            self.image_info.config(
                text=f"📏 Size: {img.size[0]}x{img.size[1]} | 📊 Format: {img.format}"
            )
            
            self.display_image(img)
            
            img_array = load_img(file_path, target_size=IMG_SIZE)
            img_array = img_to_array(img_array) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            prediction = model.predict(img_array, verbose=0)[0]
            
            top_conf = np.max(prediction) * 100
            second_conf = np.sort(prediction)[-2] * 100
            pred_std = np.std(prediction) * 100
            
            if top_conf < 40:
                self.show_invalid_result("Very low confidence - image may not be a leaf")
            elif pred_std < 2 and top_conf < 60:
                self.show_invalid_result("Ambiguous prediction - please use a clearer image")
            else:
                self.show_prediction_results(prediction)
                self.prediction_history.append(top_conf)
                self.update_stats()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to process image:\n{str(e)}")
            print(f"Error details: {e}")
    
    def display_image(self, img):
        """Display image responsively"""
        self.image_label.update()
        label_width = self.image_label.winfo_width()
        label_height = self.image_label.winfo_height()
        
        if label_width < 50 or label_height < 50:
            label_width, label_height = 300, 300
        
        img_copy = img.copy()
        img_copy.thumbnail((label_width - 20, label_height - 20), Image.Resampling.LANCZOS)
        
        photo = ImageTk.PhotoImage(img_copy)
        self.image_label.config(image=photo, text='')
        self.current_photo = photo
    
    def show_prediction_results(self, prediction):
        """Show prediction results"""
        top3_idx = np.argsort(prediction)[-3:][::-1]
        top_class = class_labels[top3_idx[0]]
        top_conf = prediction[top3_idx[0]] * 100
        
        if top_conf >= 90:
            color = COLORS['excellent']
            rating = "Excellent confidence"
        elif top_conf >= 75:
            color = COLORS['good']
            rating = "Good confidence"
        elif top_conf >= 60:
            color = COLORS['fair']
            rating = "Fair confidence"
        else:
            color = COLORS['orange']
            rating = "Low confidence"
        
        self.main_pred_label.config(
            text=f"✅ {top_class.upper()}",
            fg=color
        )
        
        self.conf_rating.config(text=rating, fg=color)
        
        if top_class in leaf_descriptions:
            self.desc_label.config(text=leaf_descriptions[top_class])
        
        self.animate_confidence_bar(top_conf, color)
        
        medals = ['🥇', '🥈', '🥉']
        for i, idx in enumerate(top3_idx):
            conf = prediction[idx] * 100
            self.top3_bars[i]['label'].config(
                text=f"{medals[i]} {class_labels[idx].upper()} - {conf:.1f}%"
            )
            self.animate_top3_bar(i, conf)
    
    def show_invalid_result(self, message):
        """Show invalid result"""
        self.main_pred_label.config(
            text="❌ INVALID IMAGE",
            fg=COLORS['red']
        )
        self.desc_label.config(
            text=f"⚠️ {message}\n\nPlease upload a clear photograph of a leaf."
        )
        self.conf_rating.config(text="", fg=COLORS['text_light'])
        self.animate_confidence_bar(0, COLORS['red'])
        
        for bar in self.top3_bars:
            bar['label'].config(text="---")
            bar['canvas'].coords(bar['bar'], 0, 0, 0, 5)
    
    def animate_confidence_bar(self, target, color):
        """Animate confidence bar"""
        def animate(current=0):
            if current < target:
                next_val = min(current + 3, target)
                width = self.conf_canvas.winfo_width()
                bar_width = int((next_val / 100) * width)
                self.conf_canvas.coords(self.conf_bar, 0, 0, bar_width, 35)
                self.conf_canvas.itemconfig(self.conf_bar, fill=color)
                
                self.conf_label.config(text=f"{int(next_val)}%", fg=color)
                
                if next_val > 15:
                    self.conf_canvas.itemconfig(self.conf_text, text=f"{int(next_val)}%")
                    self.conf_canvas.coords(self.conf_text, bar_width // 2, 17)
                else:
                    self.conf_canvas.itemconfig(self.conf_text, text="")
                
                self.root.after(15, lambda: animate(next_val))
        
        self.root.after(100, animate)
    
    def animate_top3_bar(self, index, target):
        """Animate top 3 bars"""
        def animate(current=0):
            if current < target:
                next_val = min(current + 2, target)
                width = self.top3_bars[index]['canvas'].winfo_width()
                bar_width = int((next_val / 100) * width)
                self.top3_bars[index]['canvas'].coords(
                    self.top3_bars[index]['bar'],
                    0, 0, bar_width, 5
                )
                self.top3_bars[index]['canvas'].itemconfig(
                    self.top3_bars[index]['bar'],
                    fill=self.top3_bars[index]['color']
                )
                self.root.after(10, lambda: animate(next_val))
        
        self.root.after(100 + index * 50, animate)
    
    def update_stats(self):
        """Update session statistics"""
        total = len(self.prediction_history)
        avg_conf = np.mean(self.prediction_history) if total > 0 else 0
        
        self.total_pred_label.config(text=str(total))
        self.avg_conf_label.config(text=f"{avg_conf:.1f}%")
    
    def clear_image(self):
        """Clear current image and reset UI"""
        self.current_image = None
        self.current_photo = None
        
        self.image_label.config(
            image='',
            text="🖼️\n\nNo Image Selected\n\nClick below to upload\n\nSupports: JPG, PNG, BMP"
        )
        
        self.image_info.config(text="📏 Image Size: --- | 📊 Format: ---")
        
        self.main_pred_label.config(
            text="Awaiting image upload...",
            fg=COLORS['text_light']
        )
        
        self.desc_label.config(
            text="💡 Upload a leaf image to see detailed classification and characteristics"
        )
        
        self.conf_label.config(text="0%", fg=COLORS['accent'])
        self.conf_rating.config(text="", fg=COLORS['text_light'])
        
        self.conf_canvas.coords(self.conf_bar, 0, 0, 0, 35)
        self.conf_canvas.itemconfig(self.conf_text, text="")
        
        for bar in self.top3_bars:
            bar['label'].config(text="---")
            bar['canvas'].coords(bar['bar'], 0, 0, 0, 5)

# ==============================
# MAIN EXECUTION
# ==============================
if __name__ == "__main__":
    root = tk.Tk()
    
    try:
        root.iconbitmap('leaf_icon.ico')
    except:
        pass
    
    app = EnhancedLeafGUI(root)
    root.mainloop()