"""Full connectome, new fixed local-color/cargo sensory projection, old decoder."""
import torch
from .supervised_steering import TrainableConnectome

class LayoutConnectome(TrainableConnectome):
    def __init__(self,root,initial,device='cuda',overlay=False,stock=False):
        super().__init__(root,initial,device)
        original=self.sensory_channels.clone()
        self.overlay=overlay or stock
        self.stock_sensing=stock
        self.interface='local-color-cargo-v2' if overlay else 'local-color-cargo-v1'
        if stock:self.interface='local-color-cargo-v3'
        if self.overlay:self.register_buffer('original_sensory_channels',original.clone())
        for channel in range(24):
            cells=torch.where(original==channel)[0]
            # Half retain luminance; half encode four local box-color categories
            # at the same retinal bearing. This is an engineered projection,
            # not a claim about measured biological fly color selectivity.
            for color in range(4):self.sensory_channels[cells[color::8]]=30+24*color+channel
            if stock:
                for color in range(3):self.sensory_channels[cells[color+4::8]]=129+24*color+channel
        cells=torch.where(original==27)[0]
        for cargo in range(3):self.sensory_channels[cells[cargo+1::4]]=126+cargo
        self.input_channels=201 if stock else 129
        self.fixed_hash=self.fingerprint()

    def sensory_drive(self,observations):
        if not self.overlay:return super().sensory_drive(observations)
        original=.5*observations.T[self.original_sensory_channels]
        extra=(.5 if self.stock_sensing else .25)*observations.T[self.sensory_channels]*(self.sensory_channels>=30)[:,None]
        return original+extra
