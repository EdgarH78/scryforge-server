import cv2
import numpy as np

def generate_aruco_markers():
    """Generate ArUco markers for the corners at 1 inch size in a single image"""
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    
    # 300 DPI = 300 pixels per inch
    dpi = 300
    border = dpi // 10  # 0.1 inch border
    marker_size = int((1.6 * dpi)) - (2 * border)  # 1 inch square with border
    spacing = dpi * 2  # three inch spacing from edges
    
    # Create 8.5x11 inch page at 300 DPI
    page_width = int(8.5 * dpi)
    page_height = 11 * dpi
    page = np.full((page_height, page_width, 3), 255, dtype=np.uint8)
    
    # Corner positions and IDs
    corners = [
        ('top_left', 0, spacing, spacing),
        ('top_right', 1, page_width - marker_size - spacing - 2*border, spacing),
        ('bottom_right', 2, page_width - marker_size - spacing - 2*border, page_height - marker_size - spacing - 2*border),
        ('bottom_left', 3, spacing, page_height - marker_size - spacing - 2*border)
    ]
    
    # Generate individual markers
    marker_names = ['aruco_zero.png', 'aruco_one.png', 'aruco_two.png', 'aruco_three.png']
    for marker_id, filename in enumerate(marker_names):
        # Create marker
        marker = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_size)
        
        # Add white border
        bordered = np.full((marker_size + 2*border, marker_size + 2*border), 255, dtype=np.uint8)
        bordered[border:-border, border:-border] = marker
        
        # Convert to BGR for borders and text
        bordered = cv2.cvtColor(bordered, cv2.COLOR_GRAY2BGR)
        
        # Add black border
        thickness = 2
        cv2.rectangle(bordered, (0, 0), (bordered.shape[1]-1, bordered.shape[0]-1), (0, 0, 0), thickness)
        
        # Add small text label
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = f"ID:{marker_id}"
        font_scale = 0.8
        thickness = 2
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = (bordered.shape[1] - text_size[0]) // 2
        text_y = bordered.shape[0] - 10
        
        # Draw text with white outline for better visibility
        cv2.putText(bordered, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness + 2)
        cv2.putText(bordered, text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness)
        
        # Save individual marker
        cv2.imwrite(filename, bordered)
        print(f"Generated {filename}")
    
    for name, marker_id, x, y in corners:
        # Create marker
        marker = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_size)
        
        # Add white border
        bordered = np.full((marker_size + 2*border, marker_size + 2*border), 255, dtype=np.uint8)
        bordered[border:-border, border:-border] = marker
        
        # Convert to BGR for borders and text
        bordered = cv2.cvtColor(bordered, cv2.COLOR_GRAY2BGR)
        
        # Add black border
        thickness = 2
        cv2.rectangle(bordered, (0, 0), (bordered.shape[1]-1, bordered.shape[0]-1), (0, 0, 0), thickness)
        
        # Add small text label with larger font
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = f"ID:{marker_id}"
        font_scale = 0.8
        thickness = 2
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = (bordered.shape[1] - text_size[0]) // 2
        text_y = bordered.shape[0] - 10  # Changed from -12 to -10
        
        # Draw text with white outline for better visibility
        cv2.putText(bordered, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness + 2)
        cv2.putText(bordered, text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness)
        
        # Place marker on page
        h, w = bordered.shape[:2]
        page[y:y+h, x:x+w] = bordered
    
    # Save single page with all markers
    filename = 'aruco_markers.png'
    cv2.imwrite(filename, page)
    print(f"Generated {filename} (8.5x11 inches at {dpi} DPI)")

if __name__ == "__main__":
    generate_aruco_markers()
