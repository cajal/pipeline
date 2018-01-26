""" Mask classification functions. """
import numpy as np

def classify_manual(masks, template):
    """ Opens a GUI that lets you manually classify masks into any of the valid types.

    :param np.array masks: 3-d array of masks (num_masks, image_height, image_width)
    :param np.array template: Image used as background to help with mask classification.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    mask_types= []
    plt.ioff()
    for mask in masks:
        ir = mask.sum(axis=1) > 0
        ic = mask.sum(axis=0) > 0

        il, jl = [max(np.min(np.where(i)[0]) - 10, 0) for i in [ir, ic]]
        ih, jh = [min(np.max(np.where(i)[0]) + 10, len(i)) for i in [ir, ic]]
        tmp_mask = np.array(mask[il:ih, jl:jh])

        with sns.axes_style('white'):
            fig, ax = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(10, 3))

        ax[0].imshow(template[il:ih, jl:jh], cmap=plt.cm.get_cmap('gray'))
        ax[1].imshow(template[il:ih, jl:jh], cmap=plt.cm.get_cmap('gray'))
        tmp_mask[tmp_mask == 0] = np.NaN
        ax[1].matshow(tmp_mask, cmap=plt.cm.get_cmap('viridis'), alpha=0.5, zorder=10)
        ax[2].matshow(tmp_mask, cmap=plt.cm.get_cmap('viridis'))
        for a in ax:
            a.set_aspect(1)
            a.axis('off')
        fig.tight_layout()
        fig.canvas.manager.window.wm_geometry("+250+250")
        fig.suptitle('S(o)ma, A(x)on, (D)endrite, (N)europil, (A)rtifact or (U)nknown?')

        def on_button(event):
            if event.key == 'o':
                mask_types.append('soma')
                plt.close(fig)
            elif event.key == 'x':
                mask_types.append('axon')
                plt.close(fig)
            elif event.key == 'd':
                mask_types.append('dendrite')
                plt.close(fig)
            elif event.key == 'n':
                mask_types.append('neuropil')
                plt.close(fig)
            elif event.key == 'a':
                mask_types.append('artifact')
                plt.close(fig)
            elif event.key == 'u':
                mask_types.append('unknown')
                plt.close(fig)

        fig.canvas.mpl_connect('key_press_event', on_button)

        plt.show()
    sns.reset_orig()

    return mask_types