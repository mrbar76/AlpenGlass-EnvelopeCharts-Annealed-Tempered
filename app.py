import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="AlpenGlass Window Size Visualizer",
    page_icon="🪟",
    layout="wide"
)

# Title and description
st.title("🪟 AlpenGlass Sizing Limits")

# Add comprehensive directions in collapsible expander
with st.expander("📖 How to Use This Tool - Click to expand"):
    st.markdown("""
This interactive tool helps you determine if your window dimensions fit within AlpenGlass's manufacturing capabilities for different glass configurations.

**Glass Type Selection:**
- **Tempered Glass**: Shows rectangular envelopes based on maximum long edge and short edge dimensions
- **Annealed Glass**: Shows curved envelopes based on maximum area and maximum edge length (Sizing based on wind load of DP30. Contact your sales rep if higher wind loads needed in your situation)

**Understanding the Visualization:**
- **Standard Sizing** (blue): Efficient, low-cost production range
- **Custom Range** (orange): Maximum physically achievable size (may require special order and longer lead time)
- **Minimum Size**: At least one edge must be 16" or greater
- **White areas**: Do not meet minimum size requirements

**Configuration Selection:**
- **Select "All"**: View the composite envelope showing the maximum achievable sizes across all configurations in your filter
- **Select Specific Values**: View the exact size limits for a particular glass configuration

**Checking Your Custom Size:**
1. Choose glass type (Tempered or Annealed)
2. Use the dropdowns to filter by glass specifications (or leave as "All")
3. Enter your desired width and height in the custom size input fields
4. A star will appear on the chart showing your size's location
5. Check the status indicator to see if it falls within Standard Sizing, Custom Range, or outside our capabilities

**Interpreting the Chart:**
- Hover over any point to see exact dimensions and area
- The chart displays both portrait and landscape orientations
- Download the chart as PNG to save the configuration details

**⚠️ Important Note:**
The size ranges depicted in these charts are applicable to all triple pane units and quad units with inter-pane gap >3/8". Quad units with inter-pane gap <3/8" have additional size constraints due to glass deflection risk. Talk to your sales representative if larger quad sizing is needed for your project. Engineering review required.
""")

st.markdown("---")

# Load data
@st.cache_data
def load_data():
    """Load the glass configuration data from Excel file"""
    import os
    
    possible_names = [
        'AlpenGlass max sizing data.xlsx',
        'AlpenGlass max sizing data 1.xlsx',
        'AlpenGlass_max_sizing_data.xlsx',
        'alpenglass_max_sizing_data.xlsx',
    ]
    
    for filename in possible_names:
        if os.path.exists(filename):
            try:
                tempered_df = pd.read_excel(filename, sheet_name='tempered')
                annealed_df = pd.read_excel(filename, sheet_name='annealed')
                return tempered_df, annealed_df
            except Exception as e:
                st.error(f"Error reading {filename}: {str(e)}")
                return None, None
    
    st.error("Excel file not found.")
    return None, None

def compute_envelope(rectangles, min_edge):
    """
    Compute the outer envelope (boundary) of a set of rectangles.
    Each rectangle is defined as (long_edge, short_edge).
    Returns x, y coordinates for the outer perimeter traced clockwise from origin.
    Expected path: (0,16), (16,16), (16,0), then clockwise around envelope, back to (0,16)
    """
    if not rectangles:
        return [], []
    
    # Function to compute max x at any y level  
    def max_x_at_y(y):
        max_x = 0
        for long_edge, short_edge in rectangles:
            if y <= short_edge:  # Landscape
                max_x = max(max_x, long_edge)
            if y <= long_edge:  # Portrait
                max_x = max(max_x, short_edge)
        return max_x
    
    # Get all unique y-coordinates where boundaries change
    y_values = set([0])
    for long_edge, short_edge in rectangles:
        y_values.add(short_edge)
        y_values.add(long_edge)
    y_values = sorted([y for y in y_values if y <= 150])
    
    # For each y range, determine max_x by sampling in the middle
    ranges = []
    for i in range(len(y_values) - 1):
        y_start = y_values[i]
        y_end = y_values[i + 1]
        # Sample in the middle of this range
        y_sample = y_start + 0.5
        x = max_x_at_y(y_sample)
        if x > 0:
            ranges.append((x, y_start, y_end))
    
    # Handle the last range (from last y_value to max)
    if y_values:
        y_start = y_values[-1]
        y_sample = y_start + 0.5
        x = max_x_at_y(y_sample)
        if x > 0:
            max_y_overall = max(long for long, short in rectangles)
            ranges.append((x, y_start, max_y_overall))
    
    # Build envelope path starting from (0, min_edge)
    envelope_x = [0, min_edge, min_edge]
    envelope_y = [min_edge, min_edge, 0]
    
    # Trace the right and top edges
    for i, (x, y_start, y_end) in enumerate(ranges):
        if x >= min_edge or y_end >= min_edge:
            if i == 0:
                # First range - go right along x-axis then up
                envelope_x.append(x)
                envelope_y.append(0)
                envelope_x.append(x)
                envelope_y.append(y_end)
            else:
                prev_x = ranges[i-1][0]
                if x != prev_x:
                    # Step: go up to y_start at prev_x, then left/right to new x
                    envelope_x.append(x)
                    envelope_y.append(y_start)
                
                # Go up to end of this range
                envelope_x.append(x)
                envelope_y.append(y_end)
    
    # Complete polygon back to start
    max_y = envelope_y[-1] if len(envelope_y) > 3 else min_edge
    
    # Go left to y-axis at max_y, then down to min_edge
    envelope_x.append(0)
    envelope_y.append(max_y)
    
    envelope_x.append(0)
    envelope_y.append(min_edge)
    
    return envelope_x, envelope_y

def create_tempered_plot(config_data, min_edge=16, show_all=False, all_configs_df=None, custom_point=None, filter_text="", show_labels=True):
    """Create plotly figure for tempered glass with multi-tier support"""
    
    if config_data.empty:
        return None
    
    # Collect all tiers for plotting
    if show_all and all_configs_df is not None and not all_configs_df.empty:
        # For "All" view, collect ALL actual rectangles from ALL configurations
        core_tiers = []
        tech_tiers = []
        
        for idx, row in all_configs_df.iterrows():
            # Tier 1 from this config
            core_long = row['CoreRange_ maxlongedge_inches']
            core_short = row['CoreRange_maxshortedge_inches']
            tech_long = row['Technical_limit_longedge_inches']
            tech_short = row['Technical_limit_shortedge_inches']
            
            # Add tier1 rectangles if not already in list
            core_tier1 = (core_long, core_short)
            tech_tier1 = (tech_long, tech_short)
            
            if core_tier1 not in core_tiers:
                core_tiers.append(core_tier1)
            if tech_tier1 not in tech_tiers:
                tech_tiers.append(tech_tier1)
            
            # Tier 2 if exists
            if 'CoreRange_ maxlongedge_inches_tier2' in row.index:
                if pd.notna(row['CoreRange_ maxlongedge_inches_tier2']):
                    core_tier2 = (row['CoreRange_ maxlongedge_inches_tier2'], row['CoreRange_maxshortedge_inches_tier2'])
                    if core_tier2 not in core_tiers:
                        core_tiers.append(core_tier2)
            
            if 'Technical_limit_longedge_inches_tier2' in row.index:
                if pd.notna(row['Technical_limit_longedge_inches_tier2']):
                    tech_tier2 = (row['Technical_limit_longedge_inches_tier2'], row['Technical_limit_shortedge_inches_tier2'])
                    if tech_tier2 not in tech_tiers:
                        tech_tiers.append(tech_tier2)
    else:
        # Single configuration view
        core_tiers = []
        tech_tiers = []
        
        # Tier 1 (primary)
        core_long = config_data['CoreRange_ maxlongedge_inches'].values[0]
        core_short = config_data['CoreRange_maxshortedge_inches'].values[0]
        tech_long = config_data['Technical_limit_longedge_inches'].values[0]
        tech_short = config_data['Technical_limit_shortedge_inches'].values[0]
        
        core_tiers.append((core_long, core_short))
        tech_tiers.append((tech_long, tech_short))
        
        # Tier 2 (if exists)
        if 'CoreRange_ maxlongedge_inches_tier2' in config_data.columns:
            if pd.notna(config_data['CoreRange_ maxlongedge_inches_tier2'].values[0]):
                core_long_t2 = config_data['CoreRange_ maxlongedge_inches_tier2'].values[0]
                core_short_t2 = config_data['CoreRange_maxshortedge_inches_tier2'].values[0]
                core_tiers.append((core_long_t2, core_short_t2))
        
        if 'Technical_limit_longedge_inches_tier2' in config_data.columns:
            if pd.notna(config_data['Technical_limit_longedge_inches_tier2'].values[0]):
                tech_long_t2 = config_data['Technical_limit_longedge_inches_tier2'].values[0]
                tech_short_t2 = config_data['Technical_limit_shortedge_inches_tier2'].values[0]
                tech_tiers.append((tech_long_t2, tech_short_t2))
    
    fig = go.Figure()
    
    x_range = np.arange(0, 151, 1)
    y_range = np.arange(0, 151, 1)
    X, Y = np.meshgrid(x_range, y_range)
    
    Z = np.zeros_like(X, dtype=float)
    hover_text = []
    
    for i in range(len(y_range)):
        row_text = []
        for j in range(len(x_range)):
            x, y = X[i, j], Y[i, j]
            
            meets_min = (x >= min_edge or y >= min_edge)
            
            # Check if point is in any technical tier
            in_tech = False
            for tech_long, tech_short in tech_tiers:
                if ((x <= tech_long and y <= tech_short) or 
                    (x <= tech_short and y <= tech_long)) and meets_min:
                    in_tech = True
                    break
            
            # Check if point is in any core tier
            in_core = False
            for core_long, core_short in core_tiers:
                if ((x <= core_long and y <= core_short) or 
                    (x <= core_short and y <= core_long)) and meets_min:
                    in_core = True
                    break
            
            area_sqft = (x * y) / 144
            
            if in_core:
                Z[i, j] = 2
                row_text.append(f"Width: {int(x)}\"<br>Height: {int(y)}\"<br>Area: {area_sqft:.1f} sq ft<br><b>Standard Sizing</b>")
            elif in_tech:
                Z[i, j] = 1
                row_text.append(f"Width: {int(x)}\"<br>Height: {int(y)}\"<br>Area: {area_sqft:.1f} sq ft<br><b>⚠️ Custom Range</b>")
            else:
                Z[i, j] = 0
                if not meets_min:
                    row_text.append(f"Width: {int(x)}\"<br>Height: {int(y)}\"<br>Area: {area_sqft:.1f} sq ft<br><b>Below minimum</b>")
                else:
                    row_text.append(f"Width: {int(x)}\"<br>Height: {int(y)}\"<br>Area: {area_sqft:.1f} sq ft<br><b>Outside limits</b>")
        hover_text.append(row_text)
    
    fig.add_trace(go.Heatmap(
        x=x_range, y=y_range, z=Z,
        colorscale=[[0, 'rgba(0,0,0,0)'], [1, 'rgba(0,0,0,0)']],
        showscale=False, hoverinfo='text', text=hover_text,
        hovertemplate='%{text}<extra></extra>'
    ))
    
    # Plot core range FIRST (so it appears below)
    core_labels = []
    
    if show_all and all_configs_df is not None:
        # For "All" view: plot each rectangle with fill but NO line, then add single outline
        core_all_x = []
        core_all_y = []
        
        for idx, (core_long, core_short) in enumerate(core_tiers):
            rect_x = [min_edge, core_long, core_long, core_short, core_short, 0, 0, min_edge, min_edge]
            rect_y = [0, 0, core_short, core_short, core_long, core_long, min_edge, min_edge, 0]
            
            core_all_x.extend(rect_x)
            core_all_y.extend(rect_y)
            
            if idx < len(core_tiers) - 1:
                core_all_x.append(None)
                core_all_y.append(None)
        
        # Plot filled areas WITHOUT outline
        fig.add_trace(go.Scatter(
            x=core_all_x, y=core_all_y, fill='toself',
            fillcolor='rgba(33, 150, 243, 0.3)',
            line=dict(width=0),  # No line on individual rectangles
            name='Standard Sizing',
            showlegend=True,
            hoverinfo='skip'
        ))
        
        # Compute and plot outer envelope outline
        core_envelope_x, core_envelope_y = compute_envelope(core_tiers, min_edge)
        fig.add_trace(go.Scatter(
            x=core_envelope_x, y=core_envelope_y,
            mode='lines',
            line=dict(color='rgba(33, 150, 243, 1)', width=3),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # For "All" view: only label points that are on the envelope boundary
        envelope_points = set(zip(core_envelope_x, core_envelope_y))
        for core_long, core_short in core_tiers:
            # Check if landscape corner is on envelope
            if (core_long, core_short) in envelope_points:
                label = f"{core_long}\" × {core_short}\"\n{(core_long * core_short / 144):.1f} sq ft"
                core_labels.append((core_long, core_short, label))
            # Check if portrait corner is on envelope
            if (core_short, core_long) in envelope_points:
                label = f"{core_short}\" × {core_long}\"\n{(core_short * core_long / 144):.1f} sq ft"
                core_labels.append((core_short, core_long, label))
    else:
        # Single configuration: check if multiple tiers
        if len(core_tiers) > 1:
            # Multiple tiers: plot with envelope
            core_all_x = []
            core_all_y = []
            
            for idx, (core_long, core_short) in enumerate(core_tiers):
                rect_x = [min_edge, core_long, core_long, core_short, core_short, 0, 0, min_edge, min_edge]
                rect_y = [0, 0, core_short, core_short, core_long, core_long, min_edge, min_edge, 0]
                
                core_all_x.extend(rect_x)
                core_all_y.extend(rect_y)
                
                if idx < len(core_tiers) - 1:
                    core_all_x.append(None)
                    core_all_y.append(None)
                
                core_labels.extend([
                    (core_long, core_short, f"{core_long}\" × {core_short}\"\n{(core_long * core_short / 144):.1f} sq ft"),
                    (core_short, core_long, f"{core_short}\" × {core_long}\"\n{(core_short * core_long / 144):.1f} sq ft"),
                ])
            
            # Plot filled areas WITHOUT outline
            fig.add_trace(go.Scatter(
                x=core_all_x, y=core_all_y, fill='toself',
                fillcolor='rgba(33, 150, 243, 0.3)',
                line=dict(width=0),
                name='Standard Sizing',
                showlegend=True,
                hoverinfo='skip'
            ))
            
            # Compute and plot outer envelope outline
            core_envelope_x, core_envelope_y = compute_envelope(core_tiers, min_edge)
            fig.add_trace(go.Scatter(
                x=core_envelope_x, y=core_envelope_y,
                mode='lines',
                line=dict(color='rgba(33, 150, 243, 1)', width=3),
                showlegend=False,
                hoverinfo='skip'
            ))
        else:
            # Single tier: plot normally with outline
            core_all_x = []
            core_all_y = []
            
            for idx, (core_long, core_short) in enumerate(core_tiers):
                rect_x = [min_edge, core_long, core_long, core_short, core_short, 0, 0, min_edge, min_edge]
                rect_y = [0, 0, core_short, core_short, core_long, core_long, min_edge, min_edge, 0]
                
                core_all_x.extend(rect_x)
                core_all_y.extend(rect_y)
                
                core_labels.extend([
                    (core_long, core_short, f"{core_long}\" × {core_short}\"\n{(core_long * core_short / 144):.1f} sq ft"),
                    (core_short, core_long, f"{core_short}\" × {core_long}\"\n{(core_short * core_long / 144):.1f} sq ft"),
                ])
            
            fig.add_trace(go.Scatter(
                x=core_all_x, y=core_all_y, fill='toself',
                fillcolor='rgba(33, 150, 243, 0.3)',
                line=dict(color='rgba(33, 150, 243, 1)', width=3),
                name='Standard Sizing',
                hoverinfo='skip'
            ))
    
    # Add labels for core range corners (deduplicate)
    if show_labels:
        seen_labels = set()
        for x, y, label in core_labels:
            label_key = (x, y)
            if label_key not in seen_labels:
                seen_labels.add(label_key)
                fig.add_trace(go.Scatter(
                    x=[x], y=[y],
                    mode='markers+text',
                    marker=dict(size=8, color='rgba(33, 150, 243, 0.9)', symbol='circle'),
                    text=[label],
                    textposition="top center",
                    textfont=dict(size=10, color='rgba(33, 150, 243, 1)'),
                    showlegend=False,
                    hoverinfo='skip'
                ))
    
    # Plot technical limit SECOND (so it appears on top)
    tech_labels = []
    
    if show_all and all_configs_df is not None:
        # For "All" view: plot each rectangle with fill but NO line, then add single outline
        tech_all_x = []
        tech_all_y = []
        
        for idx, (tech_long, tech_short) in enumerate(tech_tiers):
            rect_x = [min_edge, tech_long, tech_long, tech_short, tech_short, 0, 0, min_edge, min_edge]
            rect_y = [0, 0, tech_short, tech_short, tech_long, tech_long, min_edge, min_edge, 0]
            
            tech_all_x.extend(rect_x)
            tech_all_y.extend(rect_y)
            
            # Add separator between rectangles
            if idx < len(tech_tiers) - 1:
                tech_all_x.append(None)
                tech_all_y.append(None)
        
        # Plot filled areas WITHOUT outline
        fig.add_trace(go.Scatter(
            x=tech_all_x, y=tech_all_y, fill='toself',
            fillcolor='rgba(255, 152, 0, 0.2)',
            line=dict(width=0),  # No line on individual rectangles
            name='Custom Range',
            showlegend=True,
            hoverinfo='skip'
        ))
        
        # Compute and plot outer envelope outline
        tech_envelope_x, tech_envelope_y = compute_envelope(tech_tiers, min_edge)
        fig.add_trace(go.Scatter(
            x=tech_envelope_x, y=tech_envelope_y,
            mode='lines',
            line=dict(color='rgba(255, 152, 0, 0.8)', width=2, dash='dash'),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # For "All" view: only label points that are on the envelope boundary
        envelope_points = set(zip(tech_envelope_x, tech_envelope_y))
        for tech_long, tech_short in tech_tiers:
            # Check if landscape corner is on envelope
            if (tech_long, tech_short) in envelope_points:
                label = f"{tech_long}\" × {tech_short}\"\n{(tech_long * tech_short / 144):.1f} sq ft"
                tech_labels.append((tech_long, tech_short, label))
            # Check if portrait corner is on envelope
            if (tech_short, tech_long) in envelope_points:
                label = f"{tech_short}\" × {tech_long}\"\n{(tech_short * tech_long / 144):.1f} sq ft"
                tech_labels.append((tech_short, tech_long, label))
    else:
        # Single configuration: check if multiple tiers
        if len(tech_tiers) > 1:
            # Multiple tiers: plot with envelope
            tech_all_x = []
            tech_all_y = []
            
            for idx, (tech_long, tech_short) in enumerate(tech_tiers):
                rect_x = [min_edge, tech_long, tech_long, tech_short, tech_short, 0, 0, min_edge, min_edge]
                rect_y = [0, 0, tech_short, tech_short, tech_long, tech_long, min_edge, min_edge, 0]
                
                tech_all_x.extend(rect_x)
                tech_all_y.extend(rect_y)
                
                if idx < len(tech_tiers) - 1:
                    tech_all_x.append(None)
                    tech_all_y.append(None)
                
                tech_labels.extend([
                    (tech_long, tech_short, f"{tech_long}\" × {tech_short}\"\n{(tech_long * tech_short / 144):.1f} sq ft"),
                    (tech_short, tech_long, f"{tech_short}\" × {tech_long}\"\n{(tech_short * tech_long / 144):.1f} sq ft"),
                ])
            
            # Plot filled areas WITHOUT outline
            fig.add_trace(go.Scatter(
                x=tech_all_x, y=tech_all_y, fill='toself',
                fillcolor='rgba(255, 152, 0, 0.2)',
                line=dict(width=0),
                name='Custom Range',
                showlegend=True,
                hoverinfo='skip'
            ))
            
            # Compute and plot outer envelope outline
            tech_envelope_x, tech_envelope_y = compute_envelope(tech_tiers, min_edge)
            fig.add_trace(go.Scatter(
                x=tech_envelope_x, y=tech_envelope_y,
                mode='lines',
                line=dict(color='rgba(255, 152, 0, 0.8)', width=2, dash='dash'),
                showlegend=False,
                hoverinfo='skip'
            ))
        else:
            # Single tier: plot normally with outline
            tech_all_x = []
            tech_all_y = []
            
            for idx, (tech_long, tech_short) in enumerate(tech_tiers):
                rect_x = [min_edge, tech_long, tech_long, tech_short, tech_short, 0, 0, min_edge, min_edge]
                rect_y = [0, 0, tech_short, tech_short, tech_long, tech_long, min_edge, min_edge, 0]
                
                tech_all_x.extend(rect_x)
                tech_all_y.extend(rect_y)
                
                tech_labels.extend([
                    (tech_long, tech_short, f"{tech_long}\" × {tech_short}\"\n{(tech_long * tech_short / 144):.1f} sq ft"),
                    (tech_short, tech_long, f"{tech_short}\" × {tech_long}\"\n{(tech_short * tech_long / 144):.1f} sq ft"),
                ])
            
            fig.add_trace(go.Scatter(
                x=tech_all_x, y=tech_all_y, fill='toself',
                fillcolor='rgba(255, 152, 0, 0.2)',
                line=dict(color='rgba(255, 152, 0, 0.8)', width=2, dash='dash'),
                name='Custom Range',
                hoverinfo='skip'
            ))
    
    # Add labels for technical limit corners (deduplicate)
    if show_labels:
        seen_labels = set()
        for x, y, label in tech_labels:
            label_key = (x, y)
            if label_key not in seen_labels:
                seen_labels.add(label_key)
                fig.add_trace(go.Scatter(
                    x=[x], y=[y],
                    mode='markers+text',
                    marker=dict(size=8, color='rgba(255, 152, 0, 0.9)', symbol='circle'),
                    text=[label],
                    textposition="top center",
                    textfont=dict(size=10, color='rgba(255, 152, 0, 1)'),
                    showlegend=False,
                    hoverinfo='skip'
                ))
    
    if custom_point:
        add_custom_point(fig, custom_point, min_edge, core_tiers, None, tech_tiers, None, False)
    
    title_text = "AlpenGlass Sizing Limits - Tempered Glass"
    if filter_text:
        title_text += f"<br><sub>{filter_text}</sub>"
    
    fig.update_layout(
        title=dict(text=title_text, x=0.5, xanchor='center', font=dict(size=16)),
        xaxis_title="Width (inches)", yaxis_title="Height (inches)",
        xaxis=dict(range=[0, 150], showgrid=True, gridcolor='lightgray', fixedrange=True, constrain='domain'),
        yaxis=dict(range=[0, 150], showgrid=True, gridcolor='lightgray', scaleanchor="x", scaleratio=1, fixedrange=True, constrain='domain'),
        plot_bgcolor='white', hovermode='closest', height=600,
        margin=dict(l=50, r=50, t=100, b=50),
        legend=dict(orientation="v", yanchor="top", y=0.98, xanchor="right", x=0.98,
                   font=dict(size=12), bgcolor="rgba(255,255,255,0.9)", bordercolor="rgba(0,0,0,0.3)", borderwidth=1)
    )
    return fig

def generate_annealed_curve(min_edge, max_edge, max_area, is_core_range=True):
    """
    Generate the curve for annealed glass that fills BELOW the constraint curve.
    The minimum size exclusion only applies to the bottom-left corner (0-16 x 0-16).
    Returns x and y coordinates for the polygon.
    
    For 6mm core range: L-shaped boundary (95"x71" sheet constraint)
    For 5mm core range: Hybrid boundary (40 sq ft area + 95"×71" sheet constraints)
    For all other configs: Hyperbolic curve
    
    Args:
        is_core_range: If True and is 6mm or 5mm config, applies special constraints.
    """
    curve_x = []
    curve_y = []
    
    # Calculate implied short edge from area constraint
    short_edge = max_area / max_edge if max_edge > 0 else max_edge
    
    # Determine configuration type
    # 6mm: max_edge=95, max_area=6745 sq in (46.84 sq ft)
    is_6mm_config = (abs(max_edge - 95) < 1 and abs(max_area - 6745) < 10)
    # 5mm: max_edge=95, max_area=5760 sq in (40 sq ft)
    is_5mm_config = (abs(max_edge - 95) < 1 and abs(max_area - 5760) < 10)
    
    # Only apply special boundaries to core range
    apply_6mm_constraint = is_6mm_config and is_core_range
    apply_5mm_constraint = is_5mm_config and is_core_range
    
    # Start at (min_edge, 0) to avoid drawing line through excluded region
    curve_x.append(min_edge)
    curve_y.append(0)
    
    # Go up to min_edge corner
    curve_x.append(min_edge)
    curve_y.append(min_edge)
    
    # Go left to y-axis at min_edge height
    curve_x.append(0)
    curve_y.append(min_edge)
    
    # Now trace up the y-axis
    y_at_yaxis = min(max_edge, 150)
    
    # Go up the y-axis to the top of the constraint
    curve_x.append(0)
    curve_y.append(y_at_yaxis)
    
    if apply_6mm_constraint:
        # 6mm: Pure L-shaped boundary (rectangular sheet constraint only)
        # Valid region: (x ≤ 95 AND y ≤ 71) OR (x ≤ 71 AND y ≤ 95)
        
        # From y-axis, go along y=max_edge to x=short_edge
        curve_x.append(short_edge)
        curve_y.append(max_edge)
        
        # Go DOWN to the corner at (short_edge, short_edge)
        curve_x.append(short_edge)
        curve_y.append(short_edge)
        
        # Go RIGHT along y=short_edge to x=max_edge
        curve_x.append(max_edge)
        curve_y.append(short_edge)
        
        # Go DOWN to x-axis
        curve_x.append(max_edge)
        curve_y.append(0)
        
    elif apply_5mm_constraint:
        # 5mm: Hybrid boundary (area hyperbola + rectangular sheet constraint)
        # Sheet constraint: 95"×71" means short_edge_sheet = 71"
        short_edge_sheet = 71
        
        # Step 1: From (0, 95) horizontal to where hyperbola meets max_edge
        # At y=max_edge, x = max_area / max_edge
        x_at_max_edge = max_area / max_edge
        curve_x.append(x_at_max_edge)
        curve_y.append(max_edge)
        
        # Step 2: Follow hyperbola from (x_at_max_edge, max_edge) until x=short_edge_sheet
        # At x=short_edge_sheet, y = max_area / short_edge_sheet
        for x in range(int(x_at_max_edge) + 1, int(short_edge_sheet) + 1):
            y = max_area / x
            if y >= min_edge and y <= 150:
                curve_x.append(x)
                curve_y.append(y)
        
        # Step 3: At x=short_edge_sheet, add the point where hyperbola meets this x
        y_at_sheet_edge = max_area / short_edge_sheet
        curve_x.append(short_edge_sheet)
        curve_y.append(y_at_sheet_edge)
        
        # Step 4: Go straight DOWN from (short_edge_sheet, y_at_sheet_edge) to (short_edge_sheet, short_edge_sheet)
        curve_x.append(short_edge_sheet)
        curve_y.append(short_edge_sheet)
        
        # Step 5: Find where hyperbola intersects y=short_edge_sheet
        # At y=short_edge_sheet, x = max_area / short_edge_sheet
        x_at_sheet_edge = max_area / short_edge_sheet
        
        # Go RIGHT from (short_edge_sheet, short_edge_sheet) to (x_at_sheet_edge, short_edge_sheet)
        curve_x.append(x_at_sheet_edge)
        curve_y.append(short_edge_sheet)
        
        # Step 6: Follow hyperbola from (x_at_sheet_edge, short_edge_sheet) to (max_edge, y_final)
        for x in range(int(x_at_sheet_edge) + 1, int(max_edge) + 1):
            y = max_area / x
            if y >= min_edge and y <= 150:
                curve_x.append(x)
                curve_y.append(y)
        
        # Ensure we end at exactly max_edge
        y_at_max_x = max_area / max_edge
        if curve_x[-1] != max_edge:
            curve_x.append(max_edge)
            curve_y.append(y_at_max_x)
        
        # Step 7: Go DOWN to x-axis
        curve_x.append(max_edge)
        curve_y.append(0)
        
    else:
        # Original hyperbolic logic for all other configs
        x_hyperbola_at_max_edge = max_area / max_edge
        
        if x_hyperbola_at_max_edge <= max_edge:
            # Hyperbola is binding
            curve_x.append(x_hyperbola_at_max_edge)
            curve_y.append(max_edge)
            
            # Trace the hyperbola
            for x in range(int(x_hyperbola_at_max_edge) + 1, min(int(max_edge) + 1, 151)):
                y = min(max_area / x, max_edge, 150)
                
                if y >= min_edge:
                    curve_x.append(x)
                    curve_y.append(y)
                else:
                    x_at_min = max_area / min_edge
                    if x_at_min <= 150:
                        curve_x.append(x_at_min)
                        curve_y.append(min_edge)
                    break
        else:
            # Max_edge dominates
            curve_x.append(max_edge)
            curve_y.append(max_edge)
        
        # Drop down to x-axis
        if curve_x:
            last_x = curve_x[-1]
            curve_x.append(last_x)
            curve_y.append(0)
    
    # Close the polygon back to starting point (min_edge, 0)
    curve_x.append(min_edge)
    curve_y.append(0)
    
    return curve_x, curve_y

def get_annealed_label_points(min_edge, max_edge, max_area, is_core_range=True):
    """
    Get key points to label on annealed glass curves.
    Returns list of (x, y, label) tuples for key boundary points.
    
    For 6mm core range: Labels the two L-shape corners at (71, 95) and (95, 71).
    For 5mm core range: Labels hyperbolic intersections (since it's hybrid boundary).
    For all other configs: Labels where curve intersects max_edge boundaries (hyperbolic).
    """
    label_points = []
    
    # Calculate implied short edge from area constraint
    short_edge = max_area / max_edge if max_edge > 0 else max_edge
    
    # Determine configuration type
    # 6mm: max_edge=95, max_area=6745 sq in (46.84 sq ft)
    is_6mm_config = (abs(max_edge - 95) < 1 and abs(max_area - 6745) < 10)
    # 5mm: max_edge=95, max_area=5760 sq in (40 sq ft)
    is_5mm_config = (abs(max_edge - 95) < 1 and abs(max_area - 5760) < 10)
    
    # Only 6mm core gets L-shaped labeling
    apply_L_shape_labels = is_6mm_config and is_core_range
    
    if apply_L_shape_labels:
        # 6mm: Label the two corner points of the L-shaped constraint
        # Using 71" as the sheet short edge
        sheet_short_edge = 71
        
        # Point 1: (short_edge, max_edge) - portrait orientation corner
        area_sqft = (sheet_short_edge * max_edge) / 144
        label = f"{sheet_short_edge:.0f}\" × {max_edge:.0f}\"\n{area_sqft:.1f} sq ft"
        label_points.append((sheet_short_edge, max_edge, label))
        
        # Point 2: (max_edge, short_edge) - landscape orientation corner
        label = f"{max_edge:.0f}\" × {sheet_short_edge:.0f}\"\n{area_sqft:.1f} sq ft"
        label_points.append((max_edge, sheet_short_edge, label))
    else:
        # 5mm and all other configs: Original hyperbolic labeling logic
        # Point 1: Where the hyperbolic curve intersects the horizontal line y = max_edge
        x_transition = max_area / max_edge
        
        if x_transition >= min_edge and x_transition <= 150:
            y_transition = max_edge
            area_sqft = (x_transition * y_transition) / 144
            label = f"{x_transition:.0f}\" × {y_transition:.0f}\"\n{area_sqft:.1f} sq ft"
            label_points.append((x_transition, y_transition, label))
        
        # Point 2: Where the hyperbolic curve intersects the vertical line x = max_edge
        x_at_max = max_edge
        y_at_max = min(max_area / x_at_max, max_edge)
        
        if y_at_max >= min_edge and x_at_max <= 150:
            area_sqft = (x_at_max * y_at_max) / 144
            label = f"{x_at_max:.0f}\" × {y_at_max:.0f}\"\n{area_sqft:.1f} sq ft"
            label_points.append((x_at_max, y_at_max, label))
    
    return label_points

def create_annealed_plot(config_data, min_edge=16, show_all=False, all_configs_df=None, custom_point=None, filter_text="", show_labels=True):
    """Create plotly figure for annealed glass with area constraints"""
    
    if config_data.empty:
        return None
    
    if show_all and all_configs_df is not None and not all_configs_df.empty:
        core_max_edge = all_configs_df['CoreRange_maxedge_inches'].max()
        tech_max_edge = all_configs_df['Technical_limit_maxedge_inches'].max()
        core_max_area = all_configs_df['MaxArea_Core_squarefeet'].max() * 144
        tech_max_area = all_configs_df['MaxArea_Technical_limit_squarefeet'].max() * 144
    else:
        core_max_edge = config_data['CoreRange_maxedge_inches'].values[0]
        tech_max_edge = config_data['Technical_limit_maxedge_inches'].values[0]
        core_max_area = config_data['MaxArea_Core_squarefeet'].values[0] * 144
        tech_max_area = config_data['MaxArea_Technical_limit_squarefeet'].values[0] * 144
    
    fig = go.Figure()
    
    x_range = np.arange(0, 151, 1)
    y_range = np.arange(0, 151, 1)
    X, Y = np.meshgrid(x_range, y_range)
    
    Z = np.zeros_like(X, dtype=float)
    hover_text = []
    
    for i in range(len(y_range)):
        row_text = []
        for j in range(len(x_range)):
            x, y = X[i, j], Y[i, j]
            area_sqin = x * y
            area_sqft = area_sqin / 144
            meets_min = (x >= min_edge or y >= min_edge)
            max_dim = max(x, y)
            
            # Calculate implied short edges from area constraints
            tech_short_edge = tech_max_area / tech_max_edge if tech_max_edge > 0 else tech_max_edge
            core_short_edge = core_max_area / core_max_edge if core_max_edge > 0 else core_max_edge
            
            # Check if we have rectangular constraints
            # 6mm core: max_edge=95, max_area=6745 sq in (46.84 sq ft) - pure L-shape
            # 5mm core: max_edge=95, max_area=5760 sq in (40 sq ft) - hybrid (hyperbola + 95x71 sheet)
            is_6mm_core = (abs(core_max_edge - 95) < 1 and abs(core_max_area - 6745) < 10)
            is_5mm_core = (abs(core_max_edge - 95) < 1 and abs(core_max_area - 5760) < 10)
            
            tech_has_rect_constraint = False  # Never apply to tech range
            core_has_rect_constraint = is_6mm_core or is_5mm_core
            
            tech_fits_on_sheet = True
            
            core_fits_on_sheet = True
            if core_has_rect_constraint:
                # Both 5mm and 6mm are constrained by 95"×71" sheet
                sheet_short_edge = 71
                core_fits_on_sheet = ((x <= core_max_edge and y <= sheet_short_edge) or 
                                     (x <= sheet_short_edge and y <= core_max_edge))
            
            # Check technical limit constraints (area, max edge - NO rectangular constraint for tech)
            in_tech = (area_sqin <= tech_max_area and max_dim <= tech_max_edge and meets_min)
            
            # Check core range constraints (area, max edge, AND rectangular sheet for core only)
            in_core = (area_sqin <= core_max_area and max_dim <= core_max_edge and meets_min and core_fits_on_sheet)
            
            if in_core:
                Z[i, j] = 2
                row_text.append(f"Width: {int(x)}\"<br>Height: {int(y)}\"<br>Area: {area_sqft:.1f} sq ft<br><b>Standard Sizing</b>")
            elif in_tech:
                Z[i, j] = 1
                row_text.append(f"Width: {int(x)}\"<br>Height: {int(y)}\"<br>Area: {area_sqft:.1f} sq ft<br><b>⚠️ Custom Range</b>")
            else:
                Z[i, j] = 0
                if not meets_min:
                    row_text.append(f"Width: {int(x)}\"<br>Height: {int(y)}\"<br>Area: {area_sqft:.1f} sq ft<br><b>Below minimum</b>")
                else:
                    row_text.append(f"Width: {int(x)}\"<br>Height: {int(y)}\"<br>Area: {area_sqft:.1f} sq ft<br><b>Outside limits</b>")
        hover_text.append(row_text)
    
    fig.add_trace(go.Heatmap(
        x=x_range, y=y_range, z=Z,
        colorscale=[[0, 'rgba(0,0,0,0)'], [1, 'rgba(0,0,0,0)']],
        showscale=False, hoverinfo='text', text=hover_text,
        hovertemplate='%{text}<extra></extra>'
    ))
    
    # CHANGED ORDER: Plot Standard Sizing curve FIRST (so it appears below)
    core_curve_x, core_curve_y = generate_annealed_curve(min_edge, core_max_edge, core_max_area, is_core_range=True)
    
    fig.add_trace(go.Scatter(
        x=core_curve_x, y=core_curve_y, fill='toself',
        fillcolor='rgba(33, 150, 243, 0.3)',
        line=dict(color='rgba(33, 150, 243, 1)', width=3),
        name='Standard Sizing', hoverinfo='skip'
    ))
    
    # Add labels for Standard Sizing key points
    if show_labels:
        core_key_points = get_annealed_label_points(min_edge, core_max_edge, core_max_area, is_core_range=True)
        for x, y, label in core_key_points:
            fig.add_trace(go.Scatter(
                x=[x], y=[y],
                mode='markers+text',
                marker=dict(size=8, color='rgba(33, 150, 243, 0.9)', symbol='circle'),
                text=[label],
                textposition='bottom center',
                textfont=dict(size=10, color='rgb(21, 101, 192)', family='Arial'),
                showlegend=False,
                hoverinfo='skip'
            ))
    
    # CHANGED ORDER: Plot Custom Range curve SECOND (so it appears on top)
    tech_curve_x, tech_curve_y = generate_annealed_curve(min_edge, tech_max_edge, tech_max_area, is_core_range=False)
    
    fig.add_trace(go.Scatter(
        x=tech_curve_x, y=tech_curve_y, fill='toself',
        fillcolor='rgba(255, 152, 0, 0.2)',
        line=dict(color='rgba(255, 152, 0, 0.8)', width=2, dash='dash'),
        name='Custom Range', hoverinfo='skip'
    ))
    
    # Add labels for Custom Range key points
    if show_labels:
        tech_key_points = get_annealed_label_points(min_edge, tech_max_edge, tech_max_area, is_core_range=False)
        for x, y, label in tech_key_points:
            fig.add_trace(go.Scatter(
                x=[x], y=[y],
                mode='markers+text',
                marker=dict(size=8, color='rgba(255, 152, 0, 0.9)', symbol='circle'),
                text=[label],
                textposition='top center',
                textfont=dict(size=10, color='rgb(204, 102, 0)', family='Arial'),
                showlegend=False,
                hoverinfo='skip'
            ))
    
    if custom_point:
        add_custom_point(fig, custom_point, min_edge, core_max_edge, core_max_area, tech_max_edge, tech_max_area, True)
    
    title_text = "AlpenGlass Sizing Limits - Annealed Glass"
    if filter_text:
        title_text += f"<br><sub>{filter_text}</sub>"
    
    fig.update_layout(
        title=dict(text=title_text, x=0.5, xanchor='center', font=dict(size=16)),
        xaxis_title="Width (inches)", yaxis_title="Height (inches)",
        xaxis=dict(range=[0, 150], showgrid=True, gridcolor='lightgray', fixedrange=True, constrain='domain'),
        yaxis=dict(range=[0, 150], showgrid=True, gridcolor='lightgray', scaleanchor="x", scaleratio=1, fixedrange=True, constrain='domain'),
        plot_bgcolor='white', hovermode='closest', height=600,
        margin=dict(l=50, r=50, t=100, b=50),
        legend=dict(orientation="v", yanchor="top", y=0.98, xanchor="right", x=0.98,
                   font=dict(size=12), bgcolor="rgba(255,255,255,0.9)", bordercolor="rgba(0,0,0,0.3)", borderwidth=1)
    )
    return fig

def add_custom_point(fig, custom_point, min_edge, core_param1, core_param2, tech_param1, tech_param2, is_annealed):
    """Add custom size point to plot"""
    custom_width, custom_height = custom_point
    area_sqft = (custom_width * custom_height) / 144
    area_sqin = custom_width * custom_height
    meets_min = (custom_width >= min_edge or custom_height >= min_edge)
    
    if is_annealed:
        # For annealed: core_param1=max_edge, core_param2=max_area, tech_param1=max_edge, tech_param2=max_area
        max_dim = max(custom_width, custom_height)
        
        # Calculate implied short edges from area constraints
        tech_short_edge = tech_param2 / tech_param1 if tech_param1 > 0 else tech_param1
        core_short_edge = core_param2 / core_param1 if core_param1 > 0 else core_param1
        
        # Check if we have rectangular constraints
        # 6mm core: max_edge=95, max_area=6745 sq in (46.84 sq ft) - pure L-shape
        # 5mm core: max_edge=95, max_area=5760 sq in (40 sq ft) - hybrid (hyperbola + 95x71 sheet)
        is_6mm_core = (abs(core_param1 - 95) < 1 and abs(core_param2 - 6745) < 10)
        is_5mm_core = (abs(core_param1 - 95) < 1 and abs(core_param2 - 5760) < 10)
        
        tech_has_rect_constraint = False  # Never apply to tech range
        core_has_rect_constraint = is_6mm_core or is_5mm_core
        
        # Check if dimensions fit on rectangular sheets
        tech_fits_on_sheet = True
        
        core_fits_on_sheet = True
        if core_has_rect_constraint:
            # Both 5mm and 6mm are constrained by 95"×71" sheet
            sheet_short_edge = 71
            core_fits_on_sheet = ((custom_width <= core_param1 and custom_height <= sheet_short_edge) or 
                                 (custom_width <= sheet_short_edge and custom_height <= core_param1))
        
        in_tech = (area_sqin <= tech_param2 and max_dim <= tech_param1 and meets_min)
        in_core = (area_sqin <= core_param2 and max_dim <= core_param1 and meets_min and core_fits_on_sheet)
    else:
        # For tempered: core_param1=list of (long,short) tuples, core_param2=tech tiers (not used separately)
        # tech_param1=list of (long,short) tuples, tech_param2 not used
        core_tiers = core_param1
        tech_tiers = tech_param1
        
        in_tech = False
        for tech_long, tech_short in tech_tiers:
            if ((custom_width <= tech_long and custom_height <= tech_short) or 
                (custom_width <= tech_short and custom_height <= tech_long)) and meets_min:
                in_tech = True
                break
        
        in_core = False
        for core_long, core_short in core_tiers:
            if ((custom_width <= core_long and custom_height <= core_short) or 
                (custom_width <= core_short and custom_height <= core_long)) and meets_min:
                in_core = True
                break
    
    if in_core:
        marker_color, status_text = 'rgb(0, 200, 0)', "✓ Within Standard Sizing"
    elif in_tech:
        marker_color, status_text = 'rgb(255, 165, 0)', "⚠ Within Custom Range"
    elif not meets_min:
        marker_color, status_text = 'rgb(255, 0, 0)', "✗ Below Minimum Size"
    else:
        marker_color, status_text = 'rgb(255, 0, 0)', "✗ Outside Technical Limits"
    
    fig.add_trace(go.Scatter(
        x=[custom_width], y=[custom_height], mode='markers+text',
        marker=dict(size=15, color=marker_color, symbol='star', line=dict(color='white', width=2)),
        text=[f"{custom_width}\" × {custom_height}\" ({area_sqft:.1f} sf)"],
        textposition="top center",
        textfont=dict(size=12, color=marker_color, family="Arial Black"),
        name='Your Size',
        hovertemplate=f"<b>Your Custom Size</b><br>Width: {custom_width}\"<br>Height: {custom_height}\"<br>Area: {area_sqft:.1f} sq ft<br>{status_text}<extra></extra>"
    ))

def main():
    tempered_df, annealed_df = load_data()
    
    if tempered_df is None or annealed_df is None:
        st.stop()
    
    glass_type = st.radio(
        "**Select Glass Type:**",
        options=["Tempered", "Annealed"],
        horizontal=True
    )
    
    # Add checkbox to toggle label visibility
    show_labels = st.checkbox("Show dimension labels on chart", value=True)
    
    df = tempered_df if glass_type == "Tempered" else annealed_df
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        outer_lite_values = ['All'] + sorted(df['Outer Lites'].unique().tolist())
        outer_lite_labels = ['All'] + [f"{x}mm" for x in sorted(df['Outer Lites'].unique().tolist())]
        outer_lite_display = st.selectbox("Outer Lites Thickness", outer_lite_labels)
        outer_lite = 'All' if outer_lite_display == 'All' else float(outer_lite_display.replace('mm', ''))
    
    with col2:
        inner_lite_values = ['All'] + sorted(df['Inner Lite(s)'].unique().tolist())
        inner_lite_labels = ['All'] + [f"{x}mm" for x in sorted(df['Inner Lite(s)'].unique().tolist())]
        inner_lite_display = st.selectbox("Center Lite Thickness", inner_lite_labels)
        inner_lite = 'All' if inner_lite_display == 'All' else float(inner_lite_display.replace('mm', ''))
    
    st.markdown("---")
    st.markdown("### 🎯 Check Your Custom Size")
    
    size_col1, size_col2, size_col3 = st.columns([1, 1, 2])
    
    with size_col1:
        custom_width = st.number_input("Width (inches)", min_value=0.0, max_value=200.0, value=0.0, step=1.0)
    
    with size_col2:
        custom_height = st.number_input("Height (inches)", min_value=0.0, max_value=200.0, value=0.0, step=1.0)
    
    with size_col3:
        st.markdown("<br>", unsafe_allow_html=True)
        if custom_width > 0 and custom_height > 0:
            custom_area = (custom_width * custom_height) / 144
            st.info(f"**Custom Size:** {custom_width}\" × {custom_height}\" ({custom_area:.1f} sq ft)")
        else:
            st.caption("Enter dimensions to plot your custom size on the chart")
    
    st.markdown("---")
    
    # Weight calculation section
    st.markdown("### ⚖️ Thin Glass Triple Weight")
    
    # Weight lookup dictionary (lbs per square foot)
    weight_per_sqft = {
        0.5: 0.24,
        1.1: 0.53,
        1.3: 0.67,
        3: 1.64,
        4: 2.0,
        5: 2.45,
        6: 3.27
    }
    
    # Actual glass thicknesses for TPS/sealant calculations (nominal -> actual in mm)
    actual_thickness = {
        0.5: 0.5,
        1.1: 1.1,
        1.3: 1.3,
        3: 3.1,
        4: 3.9,
        5: 4.7,
        6: 5.7
    }
    
    # Calculate weight based on selected glass thicknesses
    if outer_lite != 'All' and inner_lite != 'All':
        outer_weight = weight_per_sqft.get(outer_lite, 0)
        inner_weight = weight_per_sqft.get(inner_lite, 0)
        total_weight_per_sqft = (2 * outer_weight) + inner_weight
        
        # Add OA input for TPS/sealant calculation
        oa_input_col, weight_col1, weight_col2 = st.columns([1, 1, 1])
        
        with oa_input_col:
            oa_thickness = st.number_input("OA Thickness (mm)", min_value=0.0, max_value=100.0, value=0.0, step=0.1, 
                                          help="Overall thickness of the IGU in millimeters")
        
        with weight_col1:
            st.metric("Weight per Square Foot", f"{total_weight_per_sqft:.2f} lbs/ft²")
        
        with weight_col2:
            if custom_width > 0 and custom_height > 0:
                custom_area = (custom_width * custom_height) / 144
                glass_weight = total_weight_per_sqft * custom_area
                
                # Calculate TPS and secondary sealant weight if OA is provided
                if oa_thickness > 0:
                    # Convert dimensions to cm
                    width_cm = custom_width * 2.54
                    height_cm = custom_height * 2.54
                    perimeter_cm = 2 * (width_cm + height_cm)
                    
                    # Lite thicknesses - use actual thicknesses (not nominal)
                    lite1 = actual_thickness.get(outer_lite, outer_lite)
                    lite2 = actual_thickness.get(inner_lite, inner_lite)
                    lite3 = actual_thickness.get(outer_lite, outer_lite)
                    
                    # Secondary Sealant calculation
                    # Volume = 0.4 cm × (OA - Lite1 - Lite3) cm × Perimeter cm
                    secondary_height_mm = oa_thickness - lite1 - lite3
                    secondary_height_cm = secondary_height_mm / 10  # Convert mm to cm
                    secondary_volume_cm3 = 0.4 * secondary_height_cm * perimeter_cm
                    secondary_weight_g = secondary_volume_cm3 * 1.52  # g/cm³
                    secondary_weight_lbs = secondary_weight_g / 453.592  # Convert g to lbs
                    
                    # TPS calculation
                    # Volume per cavity = 0.65 cm × [(OA - Lite1 - Lite2 - Lite3) / 2] cm × Perimeter cm
                    tps_height_per_cavity_mm = (oa_thickness - lite1 - lite2 - lite3) / 2
                    tps_height_per_cavity_cm = tps_height_per_cavity_mm / 10  # Convert mm to cm
                    tps_volume_per_cavity_cm3 = 0.65 * tps_height_per_cavity_cm * perimeter_cm
                    total_tps_volume_cm3 = tps_volume_per_cavity_cm3 * 2  # 2 cavities
                    tps_weight_g = total_tps_volume_cm3 * 1.28  # g/cm³
                    tps_weight_lbs = tps_weight_g / 453.592  # Convert g to lbs
                    
                    # Total weight
                    tps_sealant_weight = tps_weight_lbs + secondary_weight_lbs
                    total_unit_weight = glass_weight + tps_sealant_weight
                    
                    st.metric("Total Unit Weight", f"{total_unit_weight:.2f} lbs")
                    st.caption(f"<small>Glass: {glass_weight:.2f} lbs | TPS/Sealant: {tps_sealant_weight:.2f} lbs</small>", 
                              unsafe_allow_html=True)
                else:
                    st.metric("Total Unit Weight", f"{glass_weight:.2f} lbs")
                    st.caption("<small>Glass only (enter OA for TPS/sealant)</small>", unsafe_allow_html=True)
            else:
                st.metric("Total Unit Weight", "Enter dimensions")
        
        if oa_thickness > 0:
            st.caption("⚠️ TPS and secondary sealant weight is approximate")
        else:
            st.caption("⚠️ Weight is approximate and does not include weight of TPS/secondary seal")
    else:
        st.info("Select specific Outer Lites and Center Lite thicknesses to calculate weight")
    
    st.markdown("---")
    
    filtered_df = df.copy()
    
    if outer_lite != 'All':
        filtered_df = filtered_df[filtered_df['Outer Lites'] == outer_lite]
    
    if inner_lite != 'All':
        filtered_df = filtered_df[filtered_df['Inner Lite(s)'] == inner_lite]
    
    if not filtered_df.empty:
        show_all_configs = (outer_lite == 'All' or inner_lite == 'All')
        
        if show_all_configs:
            st.subheader("Size Envelope")
            config_description = []
            if outer_lite != 'All':
                config_description.append(f"Outer Lites: {outer_lite}mm")
            if inner_lite != 'All':
                config_description.append(f"Center Lite: {inner_lite}mm")
            
            if config_description:
                st.caption(f"Filtered by: {', '.join(config_description)}")
            else:
                st.caption("Showing all available configurations")
            filter_text = ", ".join(config_description) if config_description else "All Configurations"
        else:
            config_name = filtered_df['Name'].values[0]
            st.subheader(f"Configuration: {config_name}")
            filter_text = f"Configuration: {config_name}"
        
        custom_point = (custom_width, custom_height) if custom_width > 0 and custom_height > 0 else None
        
        # Use first row for plotting dimensions
        plot_data = filtered_df.iloc[[0]]
        
        plot_col, specs_col = st.columns([2, 1])
        
        with plot_col:
            if glass_type == "Tempered":
                fig = create_tempered_plot(plot_data, show_all=show_all_configs, 
                                          all_configs_df=filtered_df if show_all_configs else None, 
                                          custom_point=custom_point, filter_text=filter_text, show_labels=show_labels)
            else:
                fig = create_annealed_plot(plot_data, show_all=show_all_configs,
                                          all_configs_df=filtered_df if show_all_configs else None,
                                          custom_point=custom_point, filter_text=filter_text, show_labels=show_labels)
            
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            
            # Add disclaimer for all glass types
            st.warning("⚠️ **Important:** The size ranges depicted in these charts are applicable to all triple pane units and quad units with inter-pane gap >3/8\". Quad units with inter-pane gap <3/8\" have additional size constraints due to glass deflection risk. Talk to your sales representative if larger quad sizing is needed for your project. Engineering review required.")
            
            # Add quad sizing table
            st.markdown("#### Max Sizing for Quad Configurations with inter-pane gap < 3/8\"")
            quad_data = {
                "Outer Lites": ["3mm", "4mm", "5mm", "6mm"],
                "Max Size": ["18ft²", "25ft²", "35ft²", "40ft²"]
            }
            quad_df = pd.DataFrame(quad_data)
            st.table(quad_df)
            
            # Add annealed note
            if glass_type == "Annealed":
                st.info("**Note:** Annealed glass sizing based on wind load of DP30. Contact your sales rep if higher wind loads needed in your situation.")
        
        with specs_col:
            st.markdown("### Specifications")
            
            if glass_type == "Tempered":
                if show_all_configs:
                    # For "All" view, show the maximum dimensions found
                    core_long_max = filtered_df['CoreRange_ maxlongedge_inches'].max()
                    core_short_max = filtered_df['CoreRange_maxshortedge_inches'].max()
                    tech_long_max = filtered_df['Technical_limit_longedge_inches'].max()
                    tech_short_max = filtered_df['Technical_limit_shortedge_inches'].max()
                    
                    st.markdown("**Standard Sizing**")
                    st.info(f"Max Long Edge: **{core_long_max}\"** (across all configs)\nMax Short Edge: **{core_short_max}\"** (across all configs)")
                    
                    st.markdown("**Custom Range**")
                    st.warning(f"Max Long Edge: **{tech_long_max}\"** (across all configs)\nMax Short Edge: **{tech_short_max}\"** (across all configs)")
                else:
                    core_long_max = filtered_df['CoreRange_ maxlongedge_inches'].values[0]
                    core_short_max = filtered_df['CoreRange_maxshortedge_inches'].values[0]
                    tech_long_max = filtered_df['Technical_limit_longedge_inches'].values[0]
                    tech_short_max = filtered_df['Technical_limit_shortedge_inches'].values[0]
                    
                    st.markdown("**Standard Sizing**")
                    st.info(f"Max Long Edge: **{core_long_max}\"**\nMax Short Edge: **{core_short_max}\"**")
                    
                    st.markdown("**Custom Range**")
                    st.warning(f"Max Long Edge: **{tech_long_max}\"**\nMax Short Edge: **{tech_short_max}\"**")
            
            else:  # Annealed
                if show_all_configs:
                    core_max_edge = filtered_df['CoreRange_maxedge_inches'].max()
                    tech_max_edge = filtered_df['Technical_limit_maxedge_inches'].max()
                    core_max_area = filtered_df['MaxArea_Core_squarefeet'].max()
                    tech_max_area = filtered_df['MaxArea_Technical_limit_squarefeet'].max()
                else:
                    core_max_edge = filtered_df['CoreRange_maxedge_inches'].values[0]
                    tech_max_edge = filtered_df['Technical_limit_maxedge_inches'].values[0]
                    core_max_area = filtered_df['MaxArea_Core_squarefeet'].values[0]
                    tech_max_area = filtered_df['MaxArea_Technical_limit_squarefeet'].values[0]
                
                st.markdown("**Standard Sizing**")
                st.info(f"Max Edge: **{core_max_edge}\"**\nMax Area: **{core_max_area} sq ft**")
                
                st.markdown("**Custom Range**")
                st.warning(f"Max Edge: **{tech_max_edge}\"**\nMax Area: **{tech_max_area} sq ft**")
            
            st.markdown("**Minimum Size**")
            st.error("At least one edge must be **16\"** or greater")
            
            if custom_point:
                st.markdown("---")
                st.markdown("### 🎯 Your Custom Size Status")
                
                custom_width, custom_height = custom_point
                meets_min = (custom_width >= 16 or custom_height >= 16)
                
                if glass_type == "Tempered":
                    # Collect all tiers for checking
                    core_tiers = []
                    tech_tiers = []
                    
                    for idx, row in filtered_df.iterrows():
                        # Tier 1
                        tech_tier1 = (row['Technical_limit_longedge_inches'], row['Technical_limit_shortedge_inches'])
                        core_tier1 = (row['CoreRange_ maxlongedge_inches'], row['CoreRange_maxshortedge_inches'])
                        
                        if tech_tier1 not in tech_tiers:
                            tech_tiers.append(tech_tier1)
                        if core_tier1 not in core_tiers:
                            core_tiers.append(core_tier1)
                        
                        # Tier 2 if exists
                        if 'Technical_limit_longedge_inches_tier2' in row.index:
                            if pd.notna(row['Technical_limit_longedge_inches_tier2']):
                                tech_tier2 = (row['Technical_limit_longedge_inches_tier2'], row['Technical_limit_shortedge_inches_tier2'])
                                if tech_tier2 not in tech_tiers:
                                    tech_tiers.append(tech_tier2)
                        
                        if 'CoreRange_ maxlongedge_inches_tier2' in row.index:
                            if pd.notna(row['CoreRange_ maxlongedge_inches_tier2']):
                                core_tier2 = (row['CoreRange_ maxlongedge_inches_tier2'], row['CoreRange_maxshortedge_inches_tier2'])
                                if core_tier2 not in core_tiers:
                                    core_tiers.append(core_tier2)
                    
                    # Check against all tiers
                    in_tech = False
                    for tech_long, tech_short in tech_tiers:
                        if ((custom_width <= tech_long and custom_height <= tech_short) or 
                            (custom_width <= tech_short and custom_height <= tech_long)) and meets_min:
                            in_tech = True
                            break
                    
                    in_core = False
                    for core_long, core_short in core_tiers:
                        if ((custom_width <= core_long and custom_height <= core_short) or 
                            (custom_width <= core_short and custom_height <= core_long)) and meets_min:
                            in_core = True
                            break
                else:
                    area_sqin = custom_width * custom_height
                    max_dim = max(custom_width, custom_height)
                    
                    # Calculate implied short edges from area constraints
                    tech_short_edge = tech_max_area * 144 / tech_max_edge if tech_max_edge > 0 else tech_max_edge
                    core_short_edge = core_max_area * 144 / core_max_edge if core_max_edge > 0 else core_max_edge
                    
                    # Check if we have rectangular constraints
                    # 6mm core: max_edge=95, max_area=6745 sq in (46.84 sq ft) - pure L-shape
                    # 5mm core: max_edge=95, max_area=5760 sq in (40 sq ft) - hybrid (hyperbola + 95x71 sheet)
                    is_6mm_core = (abs(core_max_edge - 95) < 1 and abs(core_max_area * 144 - 6745) < 10)
                    is_5mm_core = (abs(core_max_edge - 95) < 1 and abs(core_max_area * 144 - 5760) < 10)
                    
                    tech_has_rect_constraint = False  # Never apply to tech range
                    core_has_rect_constraint = is_6mm_core or is_5mm_core
                    
                    # Check if dimensions fit on rectangular sheets
                    tech_fits_on_sheet = True
                    
                    core_fits_on_sheet = True
                    if core_has_rect_constraint:
                        # Both 5mm and 6mm are constrained by 95"×71" sheet
                        sheet_short_edge = 71
                        core_fits_on_sheet = ((custom_width <= core_max_edge and custom_height <= sheet_short_edge) or 
                                             (custom_width <= sheet_short_edge and custom_height <= core_max_edge))
                    
                    in_tech = (area_sqin <= tech_max_area * 144 and max_dim <= tech_max_edge and meets_min)
                    in_core = (area_sqin <= core_max_area * 144 and max_dim <= core_max_edge and meets_min and core_fits_on_sheet)
                
                if in_core:
                    st.success("✓ **Within Standard Sizing** - Standard pricing and lead time")
                elif in_tech:
                    st.warning("⚠ **Within Custom Range** - May require special order and longer lead time")
                elif not meets_min:
                    st.error("✗ **Below Minimum Size** - At least one edge must be 16\" or greater")
                else:
                    st.error("✗ **Outside Technical Limits** - This size cannot be manufactured")
    else:
        st.error("No configuration found for the selected parameters.")

if __name__ == "__main__":
    main()
