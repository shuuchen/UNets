from modules import *

class UNet(nn.Module):
	def __init__(self, n_channels, n_classes):
		super(UNet, self).__init__()
		self.inc = inconve(n_channels, 64)
		self.down1 = down(64, 128)
		self.down2 = down(128, 256)
		self.down3 = down(256, 512)
		self.down4 = down(512, 512)
		self.up1 = up(1024, 256)
		self.up2 = up(512, 128)
		self.up3 = up(256, 64)
		self.up4 = up(128, 64)
		self.outc = outconv(64, n_classes)

	# n_channels = 3, n_classes = 10 の場合、x のシェープ
	def forward(self, x):		# (h, w, 3)
		x1 = self.inc(x) 	# (h, w, 64)
		x2 = self.down1(x1) 	# (h/2, w/2, 128)
		x3 = self.down2(x2) 	# (h/4, w/4, 256)
		x4 = self.down3(x3) 	# (h/8, w/8, 512)
		x5 = self.down4(x4) 	# (h/16, w/16, 512)
		x = self.up1(x5, x4)	# (h/8, w/8, 256)
		x = self.up2(x, x3)	# (h/4, w/4, 128)
		x = self.up3(x, x2)	# (h/2, w/2, 64)
		x = self.up4(x, x1)	# (h, w, 64)
		x = self.outc(x)	# (h, w, 10)
		return NF.log_softmax(x, dim=1)
