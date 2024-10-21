import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import LassoSelector
from matplotlib import path, patches
from ipywidgets import Button, VBox, HBox, Dropdown, FloatSlider, widgets
from traitlets import link, Unicode, Bool, Any
from IPython import display
from pipeline import shared
import datajoint as dj
from datajoint.errors import DataJointError

debug_view = widgets.Output(layout={"border": "1px solid black"})


class ConfirmationButton(widgets.HBox):
    button_style = Any(default_value="")
    description = Unicode()
    disabled = Bool()
    icon = Unicode()
    layout = Any()
    style = Any()
    tooltip = Unicode()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._button = widgets.Button(**kwargs)
        self._confirm_btn = widgets.Button(
            description="confirm",
            icon="check",
            button_style="success",
            layout=dict(width="auto"),
        )
        self._cancel_btn = widgets.Button(
            description="cancel",
            icon="times",
            button_style="warning",
            layout=dict(width="auto"),
        )
        self._button.on_click(self._on_btn_click)
        self._cancel_btn.on_click(self._on_btn_click)
        self._confirm_btn.on_click(self._on_btn_click)
        self.children = [self._button]
        for key in self._button.keys:
            if key[0] != "_":
                link((self._button, key), (self, key))

    def on_click(self, *args, **kwargs):
        self._confirm_btn.on_click(*args, **kwargs)

    def _on_btn_click(self, b):
        if b == self._button:
            self.children = [self._confirm_btn, self._cancel_btn]
        else:
            self.children = [self._button]


class ManualSegmentationWidget:
    def __init__(self, key):
        # Store key for later use
        self.key = key

        # Determine which pipes are available and settings for fetching data
        # We use virtual modules here because the Tolias vs Reimer database is not consistent
        meso = dj.create_virtual_module("meso", "pipeline_meso")
        reso = dj.create_virtual_module("reso", "pipeline_reso")
        pipes = [meso, reso]
        field_tables = [pipe.ScanInfo.Field for pipe in pipes]
        field_strings = ["field", "field"]
        try:
            aod = dj.create_virtual_module("aod", "pipeline_aod")
            pipes.append(aod)
            field_tables.append(aod.ScanInfo.ROI)
            field_strings.append("roi_idx")
        except DataJointError:
            pass

        num_pipes_found = 0
        for pipe, field_table, field_string in zip(pipes, field_tables, field_strings):
            if len(pipe.SummaryImages & key) > 0:
                self.pipe = pipe
                self.field_table = field_table
                self.field_string = field_string
                num_pipes_found += 1
        if num_pipes_found == 0:
            raise ValueError("Cannot find SummaryImages for this key in any pipeline")
        elif num_pipes_found > 1:
            raise ValueError(
                "Found SummaryImages for this key in multiple pipelines. Please use a more specific key."
            )

        # Set default values for key if only one Field/ROI exists
        if "channel" not in key and (self.pipe.ScanInfo & key).fetch1("nchannels") == 1:
            self.key["channel"] = 1
        if self.field_string not in key and len(self.field_table & key) == 1:
            self.key[self.field_string] = 1

        # Download images to use for segmentation
        self.avg_image = (self.pipe.SummaryImages.Average & key).fetch1("average_image")
        self.corr_image = (self.pipe.SummaryImages.Correlation & key).fetch1(
            "correlation_image"
        )
        self.blank_image = np.zeros_like(self.avg_image)

        # Correct for negative values in the images
        self.avg_image = self.avg_image - np.nanmin(self.avg_image)
        self.corr_image = self.corr_image - np.nanmin(self.corr_image)
        self.current_image = self.avg_image

        # Create figure and axes
        self.fig, self.ax = plt.subplots(1, 1, figsize=(10, 10))
        self.fig.canvas.toolbar_visible = False
        self.fig.canvas.header_visible = False
        self.fig.canvas.footer_visible = False
        self.fig.canvas.resizable = False
        title = f"{key['animal_id']}_{key['session']}_{key['scan_idx']} Field {key[self.field_string]} Channel {key['channel']}"
        self.ax.set_title(title)

        # Create pixel coordinates used for the lasso selector
        xv, yv = np.meshgrid(
            np.arange(self.blank_image.shape[1]), np.arange(self.blank_image.shape[0])
        )
        self.pixel_coords = np.vstack((xv.flatten(), yv.flatten())).T

        # Create state variables storing masks, patches, and patch alpha to use
        self.patch_alpha = 0.5
        self.mask_array = []
        self.patch_array = []

        # Plot image
        vmin = np.min(self.avg_image[1:-1, 1:-1])
        vmax = np.max(self.avg_image[1:-1, 1:-1])
        self.imshow = self.ax.imshow(
            self.avg_image,
            origin="lower",
            interpolation="nearest",
            vmin=vmin,
            vmax=vmax,
        )

        # Create drop down menus
        self.image_menu = Dropdown(
            options=["Average Image", "Correlation Image"],
            value="Average Image",
            description="Image",
        )
        self.image_menu.observe(self.change_image, names="value")
        mask_types = shared.MaskType.fetch("type")
        self.classification_menu = Dropdown(
            options=mask_types,
            value="unknown",
            description="Mask Type",
        )

        # Create buttons for interactive segmentation
        self.delete_button = Button(description="Delete last mask")
        self.delete_button.on_click(self.delete_mask)
        self.clear_button = Button(description="Clear all masks")
        self.clear_button.on_click(self.clear_masks)
        self.close_button = Button(description="Close plot")
        self.close_button.on_click(self.close_plot)
        self.commit_button = ConfirmationButton(description="Commit Masks")
        self.commit_button.on_click(self.insert_masks)

        self.contrast_slider = FloatSlider(
            description="Contrast",
            continuous_update=False,
            min=0,
            max=50,
            value=0,
            step=0.1,
        )
        self.contrast_slider.observe(self.change_contrast, names="value")
        self.alpha_slider = FloatSlider(
            description="Mask Alpha",
            continuous_update=False,
            min=0,
            max=1,
            value=0.5,
            step=0.05,
        )
        self.alpha_slider.observe(self.change_mask_alpha, names="value")

        # Create list of GUIs to disable/enable when processing
        self.gui_list = [
            self.image_menu,
            self.delete_button,
            self.clear_button,
            self.close_button,
            self.contrast_slider,
            self.alpha_slider,
            self.classification_menu,
            self.commit_button,
        ]

        # Create lasso selector
        self.lasso = LassoSelector(self.ax, self.onselect)

        # Add a toggle button for multi-shape mode
        self.multi_shape_mode = widgets.ToggleButton(
            value=False,
            description="Multi-shape Mode",
            disabled=False,
            button_style="",  # 'success', 'info', 'warning', 'danger' or ''
            tooltip="Toggle to draw multiple shapes for the same mask",
            icon="object-group",  # FontAwesome icon
        )
        self.multi_shape_mode.observe(self.toggle_multi_shape_mode, names="value")

        # Add a variable to store the current multi-shape color
        self.current_multi_shape_color = None

        # Add a "Finish Mask" button for multi-shape mode
        self.finish_mask_button = widgets.Button(
            description="Finish Mask",
            disabled=True,
            button_style="success",
            tooltip="Click to finish the current multi-shape mask",
        )
        self.finish_mask_button.on_click(self.finish_multi_shape_mask)

        # Create a list to store temporary masks for multi-shape mode
        self.temp_mask_array = []

        # Update the GUI list to include new elements
        self.gui_list.extend([self.multi_shape_mode, self.finish_mask_button])

        # Create new gamma slider
        self.gamma_slider = FloatSlider(
            description="Gamma",
            continuous_update=False,
            min=0.1,
            max=5,
            value=1,
            step=0.1,
        )
        self.gamma_slider.observe(self.change_gamma, names="value")

        # Add gamma slider to the GUI list
        self.gui_list.append(self.gamma_slider)

        # Update the display to include the new gamma slider
        display.display(
            VBox(
                [
                    HBox([debug_view]),
                    HBox(
                        [
                            self.delete_button,
                            self.clear_button,
                            self.classification_menu,
                            self.image_menu,
                        ]
                    ),
                    HBox(
                        [
                            self.contrast_slider,
                            self.gamma_slider,
                            self.alpha_slider,
                        ]
                    ),
                    HBox([self.multi_shape_mode, self.finish_mask_button]),
                    HBox([self.commit_button, self.close_button]),
                ]
            )
        )
        plt.tight_layout()
        plt.show()

    @debug_view.capture(clear_output=True)
    def toggle_multi_shape_mode(self, change):
        if change["new"]:
            self.multi_shape_mode.button_style = "success"
            self.multi_shape_mode.icon = "check-square"
            self.finish_mask_button.disabled = False
            self.current_multi_shape_color = np.random.rand(3)
        else:
            self.multi_shape_mode.button_style = ""
            self.multi_shape_mode.icon = "object-group"
            self.finish_mask_button.disabled = True
            self.temp_mask_array = []
            self.current_multi_shape_color = None

    @debug_view.capture(clear_output=True)
    def change_image(self, dropdown):
        self.contrast_slider.value = 0
        self.gamma_slider.value = 1
        if dropdown["new"] == "Average Image":
            self.current_image = self.avg_image
        elif dropdown["new"] == "Correlation Image":
            self.current_image = self.corr_image
        self.update_image()

    @debug_view.capture(clear_output=True)
    def delete_mask(self, button):
        for gui_element in self.gui_list:
            gui_element.disabled = True

        if len(self.temp_mask_array) > 0:
            self.temp_mask_array.pop(-1)
            self.patch_array[-1].remove()  # Remove patch from plot
            self.patch_array.pop(-1)  # Remove patch from patch array
        elif len(self.mask_array) > 0:
            self.mask_array.pop(-1)
            # Remove all patches associated with the last mask if more than one
            last_mask_color = self.patch_array[-1].get_facecolor()
            while self.patch_array and np.all(
                self.patch_array[-1].get_facecolor() == last_mask_color
            ):
                self.patch_array[-1].remove()
                self.patch_array.pop(-1)
        else:
            print("No masks to delete.")

        self.fig.canvas.draw_idle()
        for gui_element in self.gui_list:
            gui_element.disabled = False

    @debug_view.capture(clear_output=True)
    def clear_masks(self, button):
        for gui_element in self.gui_list:
            gui_element.disabled = True
        self.temp_mask_array.clear()
        self.mask_array.clear()
        for patch in self.patch_array:
            patch.remove()
        self.patch_array.clear()
        self.current_multi_shape_color = None
        self.fig.canvas.draw_idle()
        for gui_element in self.gui_list:
            gui_element.disabled = False

    @debug_view.capture(clear_output=True)
    def close_plot(self, button):
        for gui_element in self.gui_list:
            gui_element.disabled = True
        plt.close()

    @debug_view.capture(clear_output=True)
    def change_contrast(self, change):
        self.update_image()

    @debug_view.capture(clear_output=True)
    def change_gamma(self, change):
        self.update_image()

    @debug_view.capture(clear_output=True)
    def update_image(self):
        # Apply gamma correction first
        gamma_corrected = np.power(self.current_image, self.gamma_slider.value)

        # Then apply contrast adjustment
        p_low, p_high = self.contrast_slider.value, 100 - self.contrast_slider.value
        vmin, vmax = np.percentile(gamma_corrected[1:-1, 1:-1], (p_low, p_high))

        # Normalize the image
        norm_image = (gamma_corrected - vmin) / (vmax - vmin)
        norm_image = np.clip(norm_image, 0, 1)

        # Update the image
        self.imshow.set_data(norm_image)
        self.imshow.set_clim(0, 1)
        self.fig.canvas.draw_idle()

    @debug_view.capture(clear_output=True)
    def change_mask_alpha(self, change):
        for gui_element in self.gui_list:
            gui_element.disabled = True
        self.patch_alpha = change["new"]
        for patch in self.patch_array:
            patch.set_alpha(self.patch_alpha)
        for gui_element in self.gui_list:
            gui_element.disabled = False

    @debug_view.capture(clear_output=True)
    def onselect(self, selection_vertices):
        for gui_element in self.gui_list:
            gui_element.disabled = True

        selection_path = path.Path(selection_vertices)
        mask_indices = selection_path.contains_points(self.pixel_coords, radius=1)
        new_mask = self.create_mask_array(mask_indices)

        if self.multi_shape_mode.value:
            self.temp_mask_array.append(new_mask)
            patch = patches.PathPatch(
                selection_path,
                facecolor=self.current_multi_shape_color,
                edgecolor="black",
                alpha=self.patch_alpha,
            )
        else:
            self.mask_array.append(new_mask)
            patch = patches.PathPatch(
                selection_path,
                facecolor=np.random.rand(3),
                edgecolor="black",
                alpha=self.patch_alpha,
            )

        self.ax.add_patch(patch)
        self.patch_array.append(patch)
        self.fig.canvas.draw_idle()
        for gui_element in self.gui_list:
            gui_element.disabled = False

    @debug_view.capture(clear_output=True)
    def finish_multi_shape_mask(self, button):
        if len(self.temp_mask_array) > 0:
            combined_mask = np.logical_or.reduce(self.temp_mask_array)
            self.mask_array.append(combined_mask)
            self.temp_mask_array = []
            print("Multi-shape mask finished and added to mask array.")
            self.current_multi_shape_color = None
        else:
            print("No shapes drawn in multi-shape mode.")

    @debug_view.capture(clear_output=True)
    def create_mask_array(self, indices):
        lin = np.arange(self.blank_image.size)
        newArray = self.blank_image.flatten()
        newArray[lin[indices]] = 1
        return newArray.reshape(self.blank_image.shape)

    @debug_view.capture(clear_output=True)
    def insert_masks(self, button):
        seg_key = {**self.key, "segmentation_method": 1}
        if len(self.mask_array) == 0 and len(self.temp_mask_array) == 0:
            raise Exception("No masks to commit. Please draw at least one mask.")

        # Finalize any ongoing multi-shape mask
        if self.multi_shape_mode.value and len(self.temp_mask_array) > 0:
            self.finish_multi_shape_mask(None)

        mask_keys = []
        for mask_id, mask in enumerate(self.mask_array, 1):
            fortran_mask = np.reshape(mask, -1, order="F").T
            fortran_mask_pixels = np.where(fortran_mask > 0)[0]
            fortran_mask_weights = np.ones_like(fortran_mask_pixels)
            mask_key = seg_key.copy()
            mask_key["pipe_version"] = 1
            mask_key["mask_id"] = mask_id
            mask_key["pixels"] = fortran_mask_pixels
            mask_key["weights"] = fortran_mask_weights
            mask_keys.append(mask_key)

        self.pipe.SegmentationTask.insert1(
            {**seg_key, "compartment": self.classification_menu.value},
            ignore_extra_fields=True,
        )
        self.pipe.Segmentation.insert1(
            {**seg_key, "pipe_version": 1},
            allow_direct_insert=True,
            ignore_extra_fields=True,
        )
        self.pipe.Segmentation.Mask.insert(mask_keys, ignore_extra_fields=True)
        self.pipe.Segmentation.Manual.insert1(
            {**seg_key, "pipe_version": 1}, ignore_extra_fields=True
        )
        self.close_plot(None)
