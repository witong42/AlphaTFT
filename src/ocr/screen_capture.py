import mss
import mss.tools
import cv2
import numpy as np
from PIL import Image
import pytesseract
from typing import Dict, List, Tuple, Optional
import logging
import os

class TFTScreenCapture:
    """Screen capture and OCR for TFT game state."""
    
    def __init__(self):
        self.sct = mss.mss()
        self.debug_dir = "debug_captures"
        os.makedirs(self.debug_dir, exist_ok=True)
        
        # Find TFT window
        self.tft_window = None
        self.find_tft_window()
        if not self.tft_window:
            print("Warning: Could not find TFT window. Using primary monitor instead.")
            self.monitor = self.sct.monitors[1]  # Primary monitor
        else:
            print(f"Found TFT window: {self.tft_window}")
            self.monitor = {
                'left': self.tft_window['x'],
                'top': self.tft_window['y'],
                'width': self.tft_window['width'],
                'height': self.tft_window['height']
            }
        
        print(f"Using monitor/window: {self.monitor}")
        
        # Take a screenshot and let user click to get coordinates
        self.calibrate_positions()
        
        # Fixed positions for 1920x1200 window
        self.base_positions = {
            'gold': {
                'x': 1650,  # Right side
                'y': 45,    # Top
                'w': 100,   # Width
                'h': 40     # Height
            },
            'shop': {
                'x': 575,   # Center
                'y': 1050,  # Bottom
                'w': 770,   # Width
                'h': 100    # Height
            },
            'bench': {
                'x': 575,   # Center
                'y': 900,   # Above shop
                'w': 770,   # Width
                'h': 100    # Height
            },
            'board': {
                'x': 575,   # Center
                'y': 300,   # Middle
                'w': 770,   # Width
                'h': 485    # Height
            },
            'items': {
                'x': 40,    # Left side
                'y': 180,   # Upper left
                'w': 190,   # Width
                'h': 760    # Height
            }
        }
    
    def find_tft_window(self):
        """Find the TFT game window."""
        try:
            # Import Quartz only on macOS
            from Quartz import (
                CGWindowListCopyWindowInfo,
                kCGWindowListOptionOnScreenOnly,
                kCGNullWindowID,
                kCGWindowLayer,
                kCGWindowAlpha
            )
            
            # Get all windows
            window_list = CGWindowListCopyWindowInfo(
                kCGWindowListOptionOnScreenOnly,
                kCGNullWindowID
            )
            
            print("\nLooking for TFT window...")
            print("Available windows:")
            
            # First pass - look for exact matches
            for window in window_list:
                name = window.get('{kCGWindowName}', '')
                owner = window.get('{kCGWindowOwnerName}', '')
                layer = window.get('{kCGWindowLayer}', 0)
                alpha = window.get('{kCGWindowAlpha}', 1.0)
                
                print(f"- Window: '{name}' (Owner: '{owner}', Layer: {layer}, Alpha: {alpha})")
                
                # First try exact match
                if ('League of Legends' in name and 'TFT' in name) or \
                   ('Teamfight Tactics' in name):
                    bounds = window.get('{kCGWindowBounds}')
                    if bounds:
                        self.tft_window = {
                            'x': int(bounds['X']),
                            'y': int(bounds['Y']),
                            'width': int(bounds['Width']),
                            'height': int(bounds['Height'])
                        }
                        print(f"\nFound TFT window with exact match:")
                        print(f"Name: {name}")
                        print(f"Owner: {owner}")
                        print(f"Bounds: {self.tft_window}")
                        return
            
            # Second pass - look for partial matches
            print("\nNo exact match found, trying partial matches...")
            for window in window_list:
                name = window.get('{kCGWindowName}', '')
                owner = window.get('{kCGWindowOwnerName}', '')
                layer = window.get('{kCGWindowLayer}', 0)
                
                # Try partial match with some validation
                if (('League of Legends' in name or 'League of Legends' in owner or
                     'TFT' in name or 'Teamfight' in name) and
                    # Ensure it's a normal window (not a menu/overlay)
                    layer == 0 and
                    # Ensure it has a reasonable size
                    window.get('{kCGWindowBounds}', {}).get('Width', 0) > 800):
                    
                    bounds = window.get('{kCGWindowBounds}')
                    if bounds:
                        self.tft_window = {
                            'x': int(bounds['X']),
                            'y': int(bounds['Y']),
                            'width': int(bounds['Width']),
                            'height': int(bounds['Height'])
                        }
                        print(f"\nFound TFT window with partial match:")
                        print(f"Name: {name}")
                        print(f"Owner: {owner}")
                        print(f"Bounds: {self.tft_window}")
                        return
            
            print("\nCould not find TFT window with any matching criteria")
            self.tft_window = None
            
        except Exception as e:
            print(f"Error finding TFT window: {e}")
            self.tft_window = None

    def calibrate_positions(self):
        """Take a screenshot and let user click to get coordinates."""
        try:
            # Capture full window
            full_screen = {
                'left': self.monitor['left'],
                'top': self.monitor['top'],
                'width': self.monitor['width'],
                'height': self.monitor['height']
            }
            
            screen_shot = self.sct.grab(full_screen)
            screen_img = np.array(screen_shot)
            
            # Convert from BGRA to BGR
            screen_img = cv2.cvtColor(screen_img, cv2.COLOR_BGRA2BGR)
            
            # Save full screenshot
            debug_path = os.path.join(self.debug_dir, "full_window.png")
            cv2.imwrite(debug_path, screen_img)
            print(f"\nSaved full window screenshot to {debug_path}")
            print("Please open this image and click on these points to get coordinates:")
            print("1. Gold amount (top right)")
            print("2. Shop (bottom center)")
            print("3. Bench (above shop)")
            print("4. Board (center)")
            print("5. Items (left side)")
            
            # For now, use these positions (adjust based on clicks)
            self.base_positions = {
                'gold': {
                    'x': 1600,  # Adjusted right
                    'y': 35,    # Top
                    'w': 120,   # Wider
                    'h': 45     # Taller
                },
                'shop': {
                    'x': 450,   # More left
                    'y': 1000,  # Adjusted down
                    'w': 1020,  # Wider
                    'h': 120    # Taller
                },
                'bench': {
                    'x': 450,   # More left
                    'y': 850,   # Adjusted down
                    'w': 1020,  # Wider
                    'h': 120    # Taller
                },
                'board': {
                    'x': 450,   # More left
                    'y': 250,   # Adjusted down
                    'w': 1020,  # Wider
                    'h': 550    # Taller
                },
                'items': {
                    'x': 20,    # More left
                    'y': 150,   # Adjusted
                    'w': 220,   # Wider
                    'h': 800    # Taller
                }
            }
            
            # Draw current regions
            debug_img = screen_img.copy()
            colors = {
                'gold': (0, 255, 0),    # Green
                'shop': (255, 0, 0),    # Blue
                'bench': (0, 0, 255),   # Red
                'board': (255, 255, 0), # Cyan
                'items': (255, 0, 255)  # Magenta
            }
            
            for region, color in colors.items():
                x = self.base_positions[region]['x']
                y = self.base_positions[region]['y']
                w = self.base_positions[region]['w']
                h = self.base_positions[region]['h']
                
                cv2.rectangle(
                    debug_img,
                    (x, y),
                    (x + w, y + h),
                    color,
                    2
                )
                cv2.putText(
                    debug_img,
                    region,
                    (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2
                )
            
            # Save debug image with regions
            debug_path = os.path.join(self.debug_dir, "regions_preview.png")
            cv2.imwrite(debug_path, debug_img)
            print(f"\nSaved regions preview to {debug_path}")
            print("Check if these regions look correct")
            
        except Exception as e:
            print(f"Error during calibration: {e}")

    def get_scaled_region(self, region: str) -> Dict:
        """Get the region bounds."""
        base = self.base_positions[region]
        
        # Use exact positions since we know the window size
        scaled = {
            'left': self.monitor['left'] + base['x'],
            'top': self.monitor['top'] + base['y'],
            'width': base['w'],
            'height': base['h']
        }
        
        return scaled
    
    def capture_screen(self, region: str) -> Optional[np.ndarray]:
        """Capture a specific region of the screen."""
        try:
            # Get the scaled region bounds
            bounds = self.get_scaled_region(region)
            
            # Capture the region
            screenshot = self.sct.grab(bounds)
            
            # Convert to numpy array
            img = np.array(screenshot)
            
            # Convert from BGRA to BGR
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            
            # Save debug image
            debug_path = os.path.join(self.debug_dir, f"{region}_capture.png")
            cv2.imwrite(debug_path, img)
            print(f"Saved {region} capture to {debug_path}")
            
            # Save a debug image of all regions
            if region == 'board':  # Only do this once
                self.save_debug_visualization()
            
            return img
        except Exception as e:
            logging.error(f"Error capturing screen region {region}: {e}")
            return None
    
    def preprocess_image(self, img: np.ndarray, region: str) -> np.ndarray:
        """Preprocess image for better OCR results."""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            if region == 'gold':
                # For gold, we want white text on black background
                # Apply threshold to make text more visible
                _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
                
                # Increase contrast
                contrast = cv2.convertScaleAbs(thresh, alpha=2.0, beta=0)
                
                # Save debug image
                debug_path = os.path.join(self.debug_dir, f"{region}_preprocessed.png")
                cv2.imwrite(debug_path, contrast)
                
                return contrast
            else:
                # For other regions, use adaptive threshold
                thresh = cv2.adaptiveThreshold(
                    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                    cv2.THRESH_BINARY, 11, 2
                )
                
                # Save debug image
                debug_path = os.path.join(self.debug_dir, f"{region}_preprocessed.png")
                cv2.imwrite(debug_path, thresh)
                
                return thresh
                
        except Exception as e:
            logging.error(f"Error preprocessing image for {region}: {e}")
            return img

    def get_gold(self) -> int:
        """Get current gold amount."""
        try:
            # Capture gold region
            img = self.capture_screen('gold')
            if img is None:
                return 0
            
            # Preprocess image
            processed = self.preprocess_image(img, 'gold')
            
            # Try to run OCR
            try:
                import pytesseract
                text = pytesseract.image_to_string(processed, config='--psm 7 -c tessedit_char_whitelist=0123456789')
            except Exception as e:
                logging.error(f"Error running Tesseract OCR: {e}")
                logging.error("Please install Tesseract OCR: brew install tesseract")
                return 0
            
            # Extract number
            numbers = ''.join(c for c in text if c.isdigit())
            if numbers:
                gold = int(numbers)
                print(f"Detected gold: {gold}")
                return gold
            else:
                print(f"No gold number found in text: {text}")
                return 0
                
        except Exception as e:
            logging.error(f"Error getting gold: {e}")
            return 0

    def detect_units(self, img: np.ndarray, region: str) -> List[Dict]:
        """Detect units in the image using template matching or OCR."""
        try:
            # Preprocess image
            processed = self.preprocess_image(img, region)
            
            # Use OCR with custom configuration for detecting shapes/units
            text = pytesseract.image_to_string(
                processed,
                config='--psm 11 --oem 1 -c tessedit_char_height_px=20'
            )
            
            print(f"Detected {region} text: {text}")
            
            # Parse detected text into unit information
            units = []
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            
            # Count distinct shapes/contours as units
            contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours by size to avoid noise
            min_area = 100  # Adjust based on your screen resolution
            valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
            
            # Create a visualization of detected units
            debug_img = img.copy()
            cv2.drawContours(debug_img, valid_contours, -1, (0,255,0), 2)
            debug_path = os.path.join(self.debug_dir, f"{region}_units.png")
            cv2.imwrite(debug_path, debug_img)
            
            # Create a unit entry for each valid contour
            for i, cnt in enumerate(valid_contours):
                x,y,w,h = cv2.boundingRect(cnt)
                units.append({
                    'character_id': f'unit_{i+1}',
                    'position': (x,y)
                })
            
            print(f"Detected {len(units)} units in {region}")
            return units
            
        except Exception as e:
            logging.error(f"Error detecting units in {region}: {e}")
            return []
            
    def detect_items(self, img: np.ndarray) -> List[Dict]:
        """Detect items in the image using template matching or OCR."""
        try:
            # Preprocess image
            processed = self.preprocess_image(img, 'items')
            
            # Use OCR with custom configuration
            text = pytesseract.image_to_string(
                processed,
                config='--psm 11'  # Sparse text with OSD
            )
            
            # Parse detected text into item information
            items = []
            for line in text.split('\n'):
                if line.strip():
                    items.append({'id': line.strip()})
            
            return items
        except Exception as e:
            logging.error(f"Error detecting items: {e}")
            return []
            
    def get_game_state(self) -> Dict:
        """Get the current game state from screen capture."""
        state = {
            'gold': 0,
            'shop': [],
            'bench': [],
            'board': [],
            'items': []
        }
        
        try:
            # Capture and process gold
            gold_img = self.capture_screen('gold')
            if gold_img is not None:
                state['gold'] = self.get_gold()
            
            # Capture and process each region
            for region in ['shop', 'bench', 'board']:
                img = self.capture_screen(region)
                if img is not None:
                    state[region] = self.detect_units(img, region)
            
            # Capture and process items
            items_img = self.capture_screen('items')
            if items_img is not None:
                state['items'] = self.detect_items(items_img)
                    
            return state
        except Exception as e:
            logging.error(f"Error getting game state: {e}")
            return state
    
    def save_debug_visualization(self):
        """Save a debug image showing all capture regions."""
        try:
            # Capture full screen
            full_screen = {
                'left': self.monitor['left'],
                'top': self.monitor['top'],
                'width': self.monitor['width'],
                'height': self.monitor['height']
            }
            
            screen_shot = self.sct.grab(full_screen)
            screen_img = np.array(screen_shot)
            
            # Draw rectangles for all regions
            debug_img = screen_img.copy()
            colors = {
                'gold': (0, 255, 0),    # Green
                'shop': (255, 0, 0),    # Blue
                'bench': (0, 0, 255),   # Red
                'board': (255, 255, 0), # Cyan
                'items': (255, 0, 255)  # Magenta
            }
            
            for region, color in colors.items():
                bounds = self.get_scaled_region(region)
                x = bounds['left'] - self.monitor['left']
                y = bounds['top'] - self.monitor['top']
                w = bounds['width']
                h = bounds['height']
                
                cv2.rectangle(
                    debug_img,
                    (x, y),
                    (x + w, y + h),
                    color,
                    2
                )
                cv2.putText(
                    debug_img,
                    region,
                    (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1
                )
            
            debug_path = os.path.join(self.debug_dir, "all_regions.png")
            cv2.imwrite(debug_path, debug_img)
            print(f"Saved regions visualization to {debug_path}")
            
        except Exception as e:
            print(f"Error saving debug visualization: {e}")
