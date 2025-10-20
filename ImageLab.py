import tkinter as tk
from tkinter import filedialog, messagebox
import os
import sys

def _resource_path(relative_path):
    base_path = getattr(sys, '_MEIPASS', None)
    if base_path:
        return os.path.join(base_path, relative_path)
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, relative_path)
try:
    from PIL import Image as _PILImage
    _PIL_AVAILABLE = True
except Exception:
    _PIL_AVAILABLE = False

class PixelImage:
    """Lightweight RGB image backed by a flat list of 8-bit values."""

    def __init__(self, width, height, pixels=None):
        """Initialize with size and optional pixel buffer.

        If ``pixels`` is omitted, the image is filled with white.
        """
        self.width = int(width)
        self.height = int(height)
        n = self.width * self.height * 3
        if pixels is not None:
            self.pixels = pixels[:]
        else:
            self.pixels = [255] * n

    def copy(self):
        """Return a deep copy of this image."""
        return PixelImage(self.width, self.height, self.pixels[:])
    @staticmethod
    def clamp(v):
        """Clamp value to 0..255 inclusive, returning an int."""
        if v < 0:
            return 0
        if v > 255:
            return 255
        return int(v)
    def index(self,x,y):
        """Compute base index for pixel (x,y)."""
        return (y*self.width + x)*3

    def get_pixel(self,x,y):
        """Fetch (r,g,b) at (x,y)."""
        i = self.index(x,y)
        r = self.pixels[i]
        g = self.pixels[i+1]
        b = self.pixels[i+2]
        return r,g,b

    def set_pixel(self,x,y,r,g,b):
        """Store (r,g,b) at (x,y), clamped to 0..255."""
        i = self.index(x,y)
        self.pixels[i]   = self.clamp(r)
        self.pixels[i+1] = self.clamp(g)
        self.pixels[i+2] = self.clamp(b)

    # PhotoImage conversions (tk.PhotoImage row-wise)
    @staticmethod
    def from_photoimage(photo):
        """Create from a tk.PhotoImage by sampling its pixels."""
        w=int(photo.width()); h=int(photo.height()); out=PixelImage(w,h)
        for y in range(h):
            for x in range(w):
                r,g,b = photo.get(x,y)
                out.set_pixel(x,y,r,g,b)
        return out
    def to_photoimage(self):
        """Convert to tk.PhotoImage for drawing on Tk canvases."""
        p=tk.PhotoImage(width=self.width, height=self.height)
        for y in range(self.height):
            base = y*self.width*3; row=[]
            for x in range(self.width):
                i=base+x*3; row.append('#%02x%02x%02x' % (self.pixels[i],self.pixels[i+1],self.pixels[i+2]))
            p.put('{'+' '.join(row)+'}', to=(0,y))
        return p

    # Histogram
    def rgb_histogram(self):
        """Return a list of 256 counts for RGB average distribution."""
        # Initialize histogram bins (0-255)
        bins = [0] * 256
        
        # Process each pixel
        for i in range(0, len(self.pixels), 3):
            # Get RGB values
            r = self.pixels[i]
            g = self.pixels[i+1]
            b = self.pixels[i+2]
            
            # Calculate average RGB value
            avg_rgb = (r + g + b) // 3
            
            # Clamp to valid range
            if avg_rgb < 0:
                avg_rgb = 0
            elif avg_rgb > 255:
                avg_rgb = 255
            
            # Increment corresponding bin
            bins[avg_rgb] += 1
        
        return bins

    # Generic pixel transform helper
    def _map_pixels(self, lut):
        """Apply a 256-entry lookup table to all pixels in-place."""
        for i in range(0, len(self.pixels), 3):
            # Apply lookup table to each RGB channel
            self.pixels[i] = lut[self.pixels[i]]      # Red channel
            self.pixels[i+1] = lut[self.pixels[i+1]]  # Green channel
            self.pixels[i+2] = lut[self.pixels[i+2]]  # Blue channel

    def to_grayscale(self):
        """In-place conversion to grayscale using simple RGB average."""
        for i in range(0, len(self.pixels), 3):
            # Get RGB values
            r = self.pixels[i]
            g = self.pixels[i+1] 
            b = self.pixels[i+2]
            
            # Calculate grayscale value using simple average
            # Grayscale = (R + G + B) / 3
            grayscale = (r + g + b) // 3
            
            # Set all channels to the same grayscale value
            self.pixels[i] = grayscale      # Red channel
            self.pixels[i+1] = grayscale    # Green channel
            self.pixels[i+2] = grayscale    # Blue channel

    def negative(self):
        """In-place photographic negative (invert channels)."""
        for i in range(0, len(self.pixels), 3):
            # Invert each RGB channel: new_value = 255 - old_value
            self.pixels[i] = 255 - self.pixels[i]      # Red channel
            self.pixels[i+1] = 255 - self.pixels[i+1]  # Green channel  
            self.pixels[i+2] = 255 - self.pixels[i+2]  # Blue channel

    def threshold(self, threshold_value):
        """In-place binary threshold on RGB average with cutoff threshold_value."""
        # Validate and clamp threshold value
        try:
            threshold_value = int(threshold_value)
        except:
            threshold_value = 128
        
        if threshold_value < 0:
            threshold_value = 0
        if threshold_value > 255:
            threshold_value = 255
        
        for i in range(0, len(self.pixels), 3):
            # Get RGB values
            r = self.pixels[i]
            g = self.pixels[i+1]
            b = self.pixels[i+2]
            
            # Calculate average RGB value
            avg_rgb = (r + g + b) // 3
            
            # Apply threshold: if average >= threshold, set to white (255), else black (0)
            if avg_rgb >= threshold_value:
                new_value = 255
            else:
                new_value = 0
            
            # Set all channels to the same value
            self.pixels[i] = new_value      # Red channel
            self.pixels[i+1] = new_value    # Green channel
            self.pixels[i+2] = new_value    # Blue channel

    def log_transform(self):
        """Logarithmic intensity mapping for contrast enhancement."""
        # Create lookup table for log transformation
        # Formula: s = c * log(1 + r) where c = 255 / log(256)
        lut = [0] * 256
        
        # Calculate scaling factor
        c = 255.0 / 5.545  # 5.545 is approximately log(256)
        
        for i in range(256):
            # Apply log transformation: s = c * log(1 + r)
            # Using natural logarithm approximation
            if i == 0:
                lut[i] = 0
            else:
                # Simple log approximation: log(1+x) ≈ x - x²/2 + x³/3 - x⁴/4
                x = i / 255.0
                log_value = x - (x*x)/2 + (x*x*x)/3 - (x*x*x*x)/4
                scaled_value = c * log_value
                lut[i] = max(0, min(255, int(scaled_value + 0.5)))
        
        # Apply the lookup table to all pixels
        self._map_pixels(lut)

    def gamma_transform(self, gamma_value):
        """Power-law (gamma) mapping for brightness adjustment."""
        # Validate gamma value
        try:
            gamma = float(gamma_value)
        except:
            gamma = 1.0
        
        if gamma <= 0:
            gamma = 1.0
        
        # Create lookup table for gamma transformation
        # Formula: s = 255 * (r/255)^gamma
        lut = [0] * 256
        
        for i in range(256):
            # Normalize input to [0,1] range
            normalized = i / 255.0
            
            # Apply gamma correction
            corrected = normalized ** gamma
            
            # Scale back to [0,255] range
            result = int(corrected * 255 + 0.5)
            
            # Clamp to valid range
            lut[i] = max(0, min(255, result))
        
        # Apply the lookup table to all pixels
        self._map_pixels(lut)

    def resize_percent(self, percent):
        """Return a new image scaled by percentage using nearest neighbor interpolation."""
        # Validate and clamp percent value
        try:
            scale = int(percent)
        except:
            scale = 100
        
        if scale < 1:
            scale = 1
        
        # Calculate new dimensions
        new_width = (self.width * scale + 50) // 100
        new_height = (self.height * scale + 50) // 100
        
        # Ensure minimum size
        if new_width < 1:
            new_width = 1
        if new_height < 1:
            new_height = 1
        
        # Create new image
        dst = PixelImage(new_width, new_height)
        
        # Resize using nearest neighbor interpolation
        for y in range(new_height):
            for x in range(new_width):
                # Calculate source pixel coordinates
                # Use integer division for nearest neighbor
                source_x = (x * self.width) // new_width
                source_y = (y * self.height) // new_height
                
                # Clamp source coordinates to valid range
                source_x = max(0, min(self.width - 1, source_x))
                source_y = max(0, min(self.height - 1, source_y))
                
                # Get pixel from source and set in destination
                r, g, b = self.get_pixel(source_x, source_y)
                dst.set_pixel(x, y, r, g, b)
        
        return dst

    def convolution(self, kernel, ksize):
        """Convolve with a square kernel of size ksize (odd)."""
        half = ksize // 2
        dst = PixelImage(self.width, self.height)
        
        for y in range(self.height):
            for x in range(self.width):
                # Initialize accumulators for RGB
                acc_r = 0.0
                acc_g = 0.0
                acc_b = 0.0
                
                # Apply kernel to surrounding pixels
                for ky in range(ksize):
                    for kx in range(ksize):
                        # Calculate source pixel coordinates
                        sx = x + kx - half
                        sy = y + ky - half
                        
                        # Check bounds
                        if sx >= 0 and sx < self.width and sy >= 0 and sy < self.height:
                            # Get kernel value
                            kernel_value = kernel[ky * ksize + kx]
                            
                            # Get pixel values
                            r, g, b = self.get_pixel(sx, sy)
                            
                            # Accumulate weighted values
                            acc_r += r * kernel_value
                            acc_g += g * kernel_value
                            acc_b += b * kernel_value
                
                # Clamp and set result pixel
                new_r = max(0, min(255, int(acc_r + 0.5)))
                new_g = max(0, min(255, int(acc_g + 0.5)))
                new_b = max(0, min(255, int(acc_b + 0.5)))
                dst.set_pixel(x, y, new_r, new_g, new_b)
        
        return dst

    def resize_to(self, new_width, new_height):
        """Return a new image resized to exactly (new_width, new_height)."""
        # Validate and clamp dimensions
        new_width = int(new_width)
        new_height = int(new_height)
        
        if new_width < 1:
            new_width = 1
        if new_height < 1:
            new_height = 1
        
        # Create new image
        dst = PixelImage(new_width, new_height)
        
        # Resize using nearest neighbor interpolation
        for y in range(new_height):
            for x in range(new_width):
                # Calculate source pixel coordinates
                # Use integer division for nearest neighbor
                source_x = (x * self.width) // new_width
                source_y = (y * self.height) // new_height
                
                # Clamp source coordinates to valid range
                source_x = max(0, min(self.width - 1, source_x))
                source_y = max(0, min(self.height - 1, source_y))
                
                # Get pixel from source and set in destination
                r, g, b = self.get_pixel(source_x, source_y)
                dst.set_pixel(x, y, r, g, b)
        
        return dst

class ImageLabApp:
    """Tkinter application wiring the UI to image operations."""

    def __init__(self,root):
        """Configure root window, initialize state, and build UI."""
        self.root=root
        self.root.title('Image Lab — Digital Image Processing')
        self.root.configure(bg='#f3f7f9')

        # Image state
        self.original_photo=None
        self.original_image=None
        self.output_image=None
        self.download_enabled=False

        # Parameter bindings
        self.kernel_size_var=tk.StringVar(value='3')
        self.threshold_var=tk.StringVar(value='128')
        self.gamma_var=tk.StringVar(value='1.0')
        self.scale_var=tk.StringVar(value='100')

        # Build UI and placeholders
        self._build_layout()
        self._init_placeholders()

    def _build_layout(self):
        """Construct static widgets and layout."""
        s=self.root
        self.scroll_canvas=tk.Canvas(s,bg='#f3f7f9',highlightthickness=0)
        self.scrollbar=tk.Scrollbar(s,orient='vertical',command=self.scroll_canvas.yview)
        self.scroll_canvas.configure(yscrollcommand=self.scrollbar.set)
        self.scroll_canvas.pack(side='left',fill='both',expand=True); self.scrollbar.pack(side='right',fill='y')
        self.content=tk.Frame(self.scroll_canvas,bg='#f3f7f9')
        self.content_id=self.scroll_canvas.create_window((0,0),window=self.content,anchor='nw')
        self.content.bind('<Configure>',self._on_content_configure); self.scroll_canvas.bind('<Configure>',self._on_canvas_configure)
        self.scroll_canvas.bind_all('<MouseWheel>',self._on_mousewheel)

        header=tk.Frame(self.content,bg='#ffffff'); header.pack(fill='x',padx=20,pady=(12,8)); header.grid_columnconfigure(0,weight=1)
        tk.Label(header,text='Image Lab — Digital Image Processing',fg='#095e66',bg='#ffffff',font=('Segoe UI',16,'bold')).grid(row=0,column=0,sticky='w')
        dev_card=tk.Frame(header,bg='#ffffff'); dev_card.grid(row=0,column=1,sticky='e')
        # developer image fixed
        try:
            dev_img_path = _resource_path('DevImg.png')
            _raw_dev=tk.PhotoImage(file=dev_img_path)
            pw,ph=_raw_dev.width(),_raw_dev.height(); target=72
            fx=max(1,pw//target); fy=max(1,ph//target); _scaled=_raw_dev.subsample(fx,fy)
            self.dev_photo=tk.PhotoImage(width=72,height=72); self._fill_rect(self.dev_photo,0,0,72,72,'#ffffff')
            for y in range(min(72,_scaled.height())):
                row=[] 
                for x in range(min(72,_scaled.width())): row.append('#%02x%02x%02x' % _scaled.get(x,y))
                self.dev_photo.put('{'+' '.join(row)+'}',to=(0,y))
        except Exception:
            self.dev_photo=tk.PhotoImage(width=72,height=72); self._fill_rect(self.dev_photo,0,0,72,72,'#dfe7ec')
        self.dev_img_label=tk.Label(dev_card,image=self.dev_photo,bg='#ffffff',bd=1,relief='solid'); self.dev_img_label.grid(row=0,column=0,padx=(0,10))
        meta=tk.Frame(dev_card,bg='#ffffff'); meta.grid(row=0,column=1)
        tk.Label(meta,text='Md. Tarakur Rahman',bg='#ffffff',fg='#0b7285',font=('Segoe UI',10,'bold')).pack(anchor='w')
        tk.Label(meta,text='ID: 0812220205101112',bg='#ffffff',fg='#6b7280',font=('Segoe UI',9)).pack(anchor='w')

        p1=tk.Frame(self.content,bg='#ffffff'); p1.pack(fill='x',padx=20,pady=8)
        grid=tk.Frame(p1,bg='#ffffff'); grid.pack(fill='both',expand=True)
        self.input_frame=tk.Frame(grid,bg='#ffffff',bd=1,relief='solid'); self.input_frame.grid(row=0,column=0,sticky='nsew',padx=(0,9),pady=9)
        tk.Label(self.input_frame,text='Input',bg='#ffffff',fg='#6b7280',font=('Segoe UI',9,'bold')).pack(anchor='w',padx=12,pady=(12,0))
        top_in=tk.Frame(self.input_frame,bg='#ffffff'); top_in.pack(anchor='ne',padx=12)
        tk.Button(top_in,text='Reset',bg='#6b7280',fg='#ffffff',font=('Segoe UI',9),relief='flat',padx=10,pady=6,command=self.reset_all).pack(side='left',padx=(0,6))
        tk.Button(top_in,text='Upload',bg='#0b7285',fg='#ffffff',font=('Segoe UI',9),relief='flat',padx=10,pady=6,command=self._choose_image).pack(side='left')
        self.input_canvas=tk.Canvas(self.input_frame,bg='#f8fafb',width=640,height=360,bd=0,highlightthickness=0); self.input_canvas.pack(padx=12,pady=12)

        self.output_frame=tk.Frame(grid,bg='#ffffff',bd=1,relief='solid'); self.output_frame.grid(row=0,column=1,sticky='nsew',padx=(9,0),pady=9)
        tk.Label(self.output_frame,text='Output',bg='#ffffff',fg='#6b7280',font=('Segoe UI',9,'bold')).pack(anchor='w',padx=12,pady=(12,0))
        top_out=tk.Frame(self.output_frame,bg='#ffffff'); top_out.pack(anchor='ne',padx=12)
        tk.Button(top_out,text='Download',bg='#0b7285',fg='#ffffff',font=('Segoe UI',9),relief='flat',padx=10,pady=6,command=self._download).pack(side='left')
        self.output_canvas=tk.Canvas(self.output_frame,bg='#f8fafb',width=640,height=360,bd=0,highlightthickness=0); self.output_canvas.pack(padx=12,pady=12)

        grid.grid_columnconfigure(0,weight=1); grid.grid_columnconfigure(1,weight=1); grid.grid_rowconfigure(0,weight=1)

        p2=tk.Frame(self.content,bg='#ffffff'); p2.pack(fill='x',padx=20,pady=8)
        self.ops_row=tk.Frame(p2,bg='#ffffff'); self.ops_row.pack(fill='x',padx=8,pady=(8,0))
        self._build_ops_buttons()
        ph=tk.Frame(p2,bg='#ffffff'); ph.pack(fill='both',expand=True,padx=8,pady=12)
        params=tk.Frame(ph,bg='#ffffff',bd=1,relief='solid'); params.grid(row=0,column=0,sticky='n',padx=(0,9))
        tk.Label(params,text='Parameters',bg='#ffffff',fg='#0b7285',font=('Segoe UI',10,'bold')).pack(anchor='w',padx=12,pady=(12,6))
        self._build_params(params)
        hist=tk.Frame(ph,bg='#ffffff',bd=1,relief='solid'); hist.grid(row=0,column=1,sticky='nsew',padx=(9,0))
        tk.Label(hist,text='Histogram',bg='#ffffff',fg='#6b7280',font=('Segoe UI',10,'bold')).pack(anchor='w',padx=12,pady=(12,6))
        self.hist_canvas=tk.Canvas(hist,bg='#fbfeff',width=560,height=200,bd=0,highlightthickness=0); self.hist_canvas.pack(fill='both',expand=True,padx=12,pady=12)
        ph.grid_columnconfigure(1,weight=1)

    def _on_content_configure(self,event):
        """Maintain scrollregion to fit content."""
        self.scroll_canvas.configure(scrollregion=self.scroll_canvas.bbox('all'))
    def _on_canvas_configure(self,event):
        """Keep content frame width in sync with canvas width."""
        self.scroll_canvas.itemconfig(self.content_id,width=event.width)
    def _on_mousewheel(self,event):
        """Mouse wheel vertical scrolling handler."""
        delta=-1*(event.delta//120); self.scroll_canvas.yview_scroll(delta,'units')

    def _build_ops_buttons(self):
        """Create operation buttons and bind their callbacks."""
        self.op_buttons=[]
        ops=[('negative','Negative'),('smoothing','Smoothing'),('sharpen','Sharpen'),
            ('resize','Resize'),('threshold','Threshold'),('log','Log Transform'),
            ('gamma','Gamma Transform'),('edge','Edge Detect'),('grayscale','Grayscale')]
        for key,label in ops:
            b=tk.Button(self.ops_row,text=label,bg='#ffffff',fg='#111827',font=('Segoe UI',9,'bold'),
                        relief='solid',bd=1,padx=10,pady=6,command=lambda k=key:self._on_op(k))
            b.pack(side='left',padx=4,pady=4); self.op_buttons.append((key,b))

    def _build_params(self,parent):
        """Build parameter input rows for operations."""
        def e(label,var):
            r=tk.Frame(parent,bg='#ffffff'); r.pack(fill='x',padx=12,pady=6)
            tk.Label(r,text=label,bg='#ffffff',fg='#6b7280',font=('Segoe UI',9)).pack(side='left')
            tk.Entry(r,textvariable=var,width=10,relief='solid',bd=1).pack(side='right')
        e('Kernel size (odd):',self.kernel_size_var); e('Threshold (0-255):',self.threshold_var)
        e('Gamma (0.1 - 5):',self.gamma_var); e('Resize %:',self.scale_var)
        tk.Label(parent,text='Tip: choose one operation at a time. Parameters reset when you press Reset.',bg='#ffffff',fg='#6b7280',font=('Segoe UI',8),wraplength=260,justify='left').pack(anchor='w',padx=12,pady=(6,12))

    def _init_placeholders(self):
        """Draw initial placeholders on canvases and histogram."""
        self._draw_placeholder(self.input_canvas,'Upload an image (Input)')
        self._draw_placeholder(self.output_canvas,'Output will appear here')
        self._draw_histogram_empty(); self._set_ops_enabled(True); self._clear_active_ops(); self.download_enabled=False

    def _draw_placeholder(self,canvas,text):
        """Render a neutral placeholder state on a given canvas."""
        w=int(canvas.winfo_reqwidth()); h=int(canvas.winfo_reqheight()); canvas.delete('all')
        canvas.create_rectangle(0,0,w,h,fill='#f6fafb',width=0); canvas.create_text(w//2,h//2,text=text,fill='#9aa6ae',font=('Segoe UI',12))

    def _draw_histogram_empty(self):
        """Draw the empty histogram placeholder panel."""
        c=self.hist_canvas; w=int(c.winfo_reqwidth()); h=int(c.winfo_reqheight()); c.delete('all')
        c.create_rectangle(0,0,w,h,fill='#f3f7f9',width=0); c.create_text(w//2,h//2,text='Histogram (empty)',fill='#9aa6ae',font=('Segoe UI',12))

    def _on_op(self,key):
        """Handle operation button click and schedule processing."""
        if self.original_image is None:
            messagebox.showwarning('Info','Please upload an image first.'); return
        self._clear_active_ops(); self._set_active_button(key); self._set_ops_enabled(False)
        self.root.after(10,lambda:self._apply_operation(key))

    def _set_active_button(self,key):
        """Apply active styling to the selected operation button."""
        for k,b in self.op_buttons:
            if k==key: b.configure(bg='#0b7285',fg='#ffffff')
    def _clear_active_ops(self):
        """Clear active styling from all operation buttons."""
        for _,b in self.op_buttons: b.configure(bg='#ffffff',fg='#111827')
    def _set_ops_enabled(self,enabled):
        """Enable/disable all operation buttons together."""
        for _,b in self.op_buttons: b.configure(state='normal' if enabled else 'disabled')

    def _apply_operation(self,key):
        """Apply selected image operation and refresh output/histogram."""
        try:
            # Use output image if it exists (for chaining), otherwise use original image
            base = self.output_image.copy() if self.output_image is not None else self.original_image.copy()
            
            if key=='negative':
                base.negative(); self.output_image=base
            elif key=='smoothing':
                k=self._safe_int(self.kernel_size_var.get(),3); k=k+1 if k%2==0 else k
                size=k*k; v=1.0/float(size); kernel=[v]*size; self.output_image=base.convolution(kernel,k)
            elif key=='sharpen':
                self.output_image=base.convolution([0,-1,0,-1,5,-1,0,-1,0],3)
            elif key=='resize':
                p=self._safe_int(self.scale_var.get(),100); self.output_image=base.resize_percent(p)
            elif key=='threshold':
                t=self._safe_int(self.threshold_var.get(),128); base.threshold(t); self.output_image=base
            elif key=='log':
                base.log_transform(); self.output_image=base
            elif key=='gamma':
                g=self._safe_float(self.gamma_var.get(),1.0); base.gamma_transform(g); self.output_image=base
            elif key=='edge':
                self.output_image=base.convolution([-1,-1,-1,-1,8,-1,-1,-1,-1],3)
            elif key=='grayscale':
                base.to_grayscale(); self.output_image=base
            else: self.output_image=base
            
            # Enable download after any operation
            self.download_enabled=True
            self._render_to_canvases(); self._update_histogram()
        except Exception:
            messagebox.showerror('Error','An error occurred while processing the image.')
        finally:
            self._set_ops_enabled(True)

    def _choose_image(self):
        """Open a file dialog and load the chosen image into the app."""
        path=filedialog.askopenfilename(title='Choose image',filetypes=[
            ('Images','*.png *.gif *.ppm *.pgm *.pbm *.bmp *.jpg *.jpeg *.tif *.tiff *.webp'),
            ('PNG','*.png'),('GIF','*.gif'),('PPM/PGM/PBM','*.ppm *.pgm *.pbm'),('BMP','*.bmp'),
            ('JPEG','*.jpg *.jpeg'),('TIFF','*.tif *.tiff'),('WEBP','*.webp'),('All files','*.*')])
        if not path: return
        photo=None
        try:
            photo=tk.PhotoImage(file=path)
        except Exception:
            photo=None
        if photo is None and _PIL_AVAILABLE:
            try:
                pil=_PILImage.open(path).convert('RGB'); w,h=pil.size
                pix=list(pil.tobytes()); pix_img=PixelImage(w,h,pix); photo=pix_img.to_photoimage()
            except Exception:
                photo=None
        if photo is None:
            ext = path.lower().rsplit('.',1)[-1] if '.' in path else ''
            if ext == 'bmp':
                try:
                    bmp_img=self._load_bmp24(path); photo=bmp_img.to_photoimage()
                except Exception:
                    photo=None
        if photo is None:
            messagebox.showerror('Error','Unable to load that image file on this Tk build. Please use PNG/GIF/BMP.')
            return
        max_dim=1024; pw=photo.width(); ph=photo.height()
        if pw>max_dim or ph>max_dim:
            factor=max((pw+max_dim-1)//max_dim,(ph+max_dim-1)//max_dim)
            try: photo=photo.subsample(factor,factor)
            except Exception: pass
        self.original_photo=photo
        orig=PixelImage.from_photoimage(photo)
        cw=max(self.input_canvas.winfo_width(),self.input_canvas.winfo_reqwidth())
        ch=max(self.input_canvas.winfo_height(),self.input_canvas.winfo_reqheight())
        if orig.width>0 and orig.height>0:
            scale_w=cw/float(orig.width); scale_h=ch/float(orig.height); scale=scale_w if scale_w<scale_h else scale_h
            target_w=int(orig.width*scale); target_h=int(orig.height*scale)
            if target_w<1: target_w=1
            if target_h<1: target_h=1
            fitted=orig.resize_to(target_w,target_h)
        else:
            fitted=orig
        self.original_image=fitted; self.output_image=None
        self._render_to_canvases(); self._update_histogram(); self._clear_active_ops(); self._reset_params(False); self.download_enabled=False

    def _choose_dev_photo(self):
        """Notify that the developer photo is fixed and cannot be changed."""
        messagebox.showinfo('Info','Developer image is fixed for this app.')

    def _render_to_canvases(self):
        """Render input and output images to their respective canvases."""
        if self.original_image is None: return
        in_photo=self.original_image.to_photoimage(); self._draw_image_centered(self.input_canvas,in_photo); self.input_canvas.image=in_photo
        if self.output_image is None:
            self._draw_placeholder(self.output_canvas,'Output will appear here')
            if hasattr(self.output_canvas,'image'):
                delattr(self.output_canvas,'image')
        else:
            out_photo=self.output_image.to_photoimage(); self._draw_image_centered(self.output_canvas,out_photo); self.output_canvas.image=out_photo

    def _draw_image_centered(self,canvas,photo):
        """Center image on canvas, downscaling if larger than viewport."""
        canvas.delete('all')
        cw=max(canvas.winfo_width(),canvas.winfo_reqwidth()); ch=max(canvas.winfo_height(),canvas.winfo_reqheight())
        canvas.create_rectangle(0,0,cw,ch,fill='#ffffff',width=0)
        pw=photo.width(); ph=photo.height()
        if pw>0 and ph>0 and (pw>cw or ph>ch):
            scale_w=cw/float(pw); scale_h=ch/float(ph); scale=scale_w if scale_w<scale_h else scale_h
            tw=int(pw*scale); th=int(ph*scale)
            if tw<1: tw=1
            if th<1: th=1
            img=PixelImage.from_photoimage(photo); img=img.resize_to(tw,th); photo=img.to_photoimage(); canvas.image_tmp=photo
        canvas.create_image(cw//2,ch//2,image=photo)

    def _update_histogram(self):
        """Recompute luminance histogram and draw bars."""
        c=self.hist_canvas; c.delete('all')
        w=max(c.winfo_width(),c.winfo_reqwidth()); h=max(c.winfo_height(),c.winfo_reqheight())
        c.create_rectangle(0,0,w,h,fill='#fbfeff',width=0)
        
        # Use output image if available, otherwise use original image
        image_to_hist = self.output_image if self.output_image is not None else self.original_image
        
        if image_to_hist is None:
            self._draw_histogram_empty(); return
            
        bins=image_to_hist.rgb_histogram()
        max_bin=1
        for v in bins:
            if v>max_bin: max_bin=v
        bar_w=float(w)/256.0
        for i,v in enumerate(bins):
            bh=int(h*(v/float(max_bin))) if max_bin>0 else 0
            x0=int(i*bar_w); x1=int((i+1)*bar_w)
            c.create_rectangle(x0,h-bh,x1,h,fill='#0b7285',width=0)

    def reset_all(self):
        """Restore original image and default parameters, clearing all operations."""
        if self.original_image is None:
            self._init_placeholders(); return
        # Clear all operations and reset to original image
        self.output_image=None; self._reset_params(False); self._clear_active_ops()
        self._render_to_canvases(); self._update_histogram(); self._set_ops_enabled(True); self.download_enabled=False

    def _reset_params(self,redraw=True):
        """Reset parameter fields to default values and optionally redraw."""
        self.kernel_size_var.set('3'); self.threshold_var.set('128'); self.gamma_var.set('1.0'); self.scale_var.set('100')
        if redraw and self.original_image is not None: self._render_to_canvases(); self._update_histogram()

    def _download(self):
        """Save the current output image to disk in PNG or PPM format."""
        if not self.download_enabled or self.output_image is None:
            messagebox.showwarning('Info','No image to download.'); return
        path=filedialog.asksaveasfilename(title='Save image',defaultextension='.png',filetypes=[('PNG','*.png'),('PPM','*.ppm')])
        if not path: return
        try:
            photo=self.output_image.to_photoimage()
            try:
                photo.write(path,format='png'); return
            except Exception:
                if not path.lower().endswith('.ppm'): path=path+'.ppm'
                self._save_ppm(path,self.output_image)
        except Exception:
            try:
                if not path.lower().endswith('.ppm'): path=path+'.ppm'
                self._save_ppm(path,self.output_image)
            except Exception:
                messagebox.showerror('Error','Failed to save the image.')

    def _save_ppm(self,path,img):
        """Write image in binary P6 PPM format to the given path."""
        with open(path,'wb') as f:
            header='P6\n{} {}\n255\n'.format(img.width,img.height)
            f.write(header.encode('ascii')); f.write(bytes(img.pixels))

    def _fill_rect(self,photo,x,y,w,h,color):
        """Fill a rectangle on a PhotoImage with a hex color string."""
        for yy in range(y,y+h):
            row=[color]*w; photo.put('{'+' '.join(row)+'}',to=(x,yy))

    def _safe_int(self,value,default):
        """Safely parse integer, returning default on error."""
        try: return int(value)
        except Exception: return default
    def _safe_float(self,value,default):
        """Safely parse float, returning default on error."""
        try: return float(value)
        except Exception: return default

    def _load_bmp24(self,path):
        """Load a 24-bit uncompressed BMP file into PixelImage."""
        with open(path,'rb') as f: data=f.read()
        if len(data)<54 or data[0:2]!=b'BM': raise Exception('Not a BMP file')
        dib_size=int.from_bytes(data[14:18],'little')
        if dib_size<40: raise Exception('Unsupported BMP header')
        w=int.from_bytes(data[18:22],'little'); h=int.from_bytes(data[22:26],'little')
        planes=int.from_bytes(data[26:28],'little'); bpp=int.from_bytes(data[28:30],'little'); comp=int.from_bytes(data[30:34],'little')
        if planes!=1 or bpp!=24 or comp!=0: raise Exception('Only 24-bit uncompressed BMP supported')
        pixel_offset=int.from_bytes(data[10:14],'little')
        row_stride=((w*3+3)//4)*4
        pixels=[0]*(w*h*3)
        for y in range(h):
            row_start=pixel_offset + (h-1-y)*row_stride
            for x in range(w):
                idx=row_start + x*3
                b=data[idx]; g=data[idx+1]; r=data[idx+2]
                pi=(y*w+x)*3; pixels[pi]=r; pixels[pi+1]=g; pixels[pi+2]=b
        return PixelImage(w,h,pixels)

def main():
    """Application entry point: create Tk root and run mainloop."""
    root=tk.Tk(); app=ImageLabApp(root); root.minsize(960,600); root.mainloop()

if __name__=='__main__':
    main()
