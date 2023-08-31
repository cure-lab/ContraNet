from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from advertorch.utils import calc_l2distsq
from advertorch.utils import tanh_rescale
from advertorch.utils import torch_arctanh
from advertorch.utils import clamp
from advertorch.utils import to_one_hot
from advertorch.utils import replicate_input

from cw_base import Attack
from cw_base import LabelMixin

from cw_utils import is_successful
import lib.pytorch_ssim as ssim_package

CARLINI_L2DIST_UPPER = 1e10
CARLINI_LinfDIST_UPPER = 1
CARLINI_COEFF_UPPER = 1e10
INVALID_LABEL= -1
REPEAT_STEP = 10
ONE_MINUS_EPS = 0.999999
UPPER_CHECK = 1e9
PREV_LOSS_INIT = 1e6
TARGET_MULT = 10000.0
NUM_CHECKS = 10

c_con = 0.
mlploss = nn.CrossEntropyLoss()
ssim_size_average = lambda x, y: ssim_package.ssim(x, y, size_average=False)
mean = [0.4914, 0.4822, 0.4465]
std = [0.2023, 0.1994, 0.2010]
mean =torch.tensor(mean, dtype=torch.float32).cuda()
std =  torch.tensor(std, dtype=torch.float32).cuda()
mean = mean[:,None, None]
std = std[:, None, None]


def normalize(X):
    return (X - mean)/std


class CarliniWagnerL2Attack(Attack, LabelMixin):
	"""
	The Carlini Wagner L2 Attack,
	:param predict: forward pass function.
	:param num_classes: number of classes.
	:param targeted: if the attack is targeted.
	:param learning_rate: the learning rate for the attack algorithm.
	:param binary_search_steps: number of binary search times to find the optimum
	:param max_iterations: the maximum number of iterations
	:param abort_early: if set to true, abort early if getting stuck in local
	:param initial_const: intial value of the constant c
	:param clip_min: minmum value per input dimension.
	:param clip_max: maximum value per input dimension.
	"""

	def __init__(self, predict, 
				
				encoder, vae, gen, orimodel, genmodel, mlpModel, discriminator, batchsize=128,
				
				num_classes=10, confidence=0, targeted=False, learning_rate=0.01,
				binary_search_steps=9, max_iterations=10000,
				abort_early=True, initial_const=1e-3,
				clip_min = 0., clip_max=1., loss_fn=None,normalize_fn=None,
				adaptive_evi=False, evi_train_median=None,
				adaptive_con=False, con_train_median=None, adaptive_loss="all"):
				
		loss_fn = None
		super(CarliniWagnerL2Attack, self).__init__(predict, loss_fn, clip_min, clip_max)

		self.learning_rate = learning_rate
		self.max_iterations = max_iterations
		self.binary_search_steps = binary_search_steps
		self.abort_early = abort_early
		self.confidence = confidence
		self.initial_const = initial_const
		self.num_classes = num_classes
		# The last iteration (if we run many steps) repeat the search once.
		self.repeat = binary_search_steps >= REPEAT_STEP#False
		self.targeted = targeted
		self.normalize_fn = normalize_fn

		self.adaptive_evi = adaptive_evi
		self.evi_train_median = evi_train_median
		self.adaptive_con = adaptive_con
		self.con_train_median = con_train_median

		#contraNet yyj
		self.encoder = encoder
		self.vae = vae
		self.gen = gen
		self.discriminator = discriminator
		self.orimodel = orimodel
		self.genmodel = genmodel
		self.mlpModel = mlpModel
		self.batchsize = batchsize
		# print("batchsize is:", batchsize)
		self.dis_decision = torch.zeros(batchsize)
		self.ssim_decision = torch.zeros(batchsize)
		# self.l2_decision = torch.zeros(batchsize)
		self.dml_decision = torch.zeros(batchsize)
		self.adaptive_loss = adaptive_loss
		

	def _loss_fn(self, output, y_onehot, l2distsq, const):
		# TODO: move this out of the class and make this the default loss_fn
		#   after having targeted tests implemented
		real = (y_onehot * output).sum(dim=1)

		# TODO: make loss modular, write a loss class
		other, label_o = ((1.0 - y_onehot) * output - (y_onehot * TARGET_MULT)#Target_mult=10000.0
				 ).max(1)

		# confidence, label max(1) means the largest element in row of a matrix
		label_0 = F.one_hot(label_o, num_classes=self.num_classes) 
		# - (y_onehot * TARGET_MULT) is for the true label not to be selected

		if self.adaptive_con:
			c = c_con #default=0
		else:
			c = self.confidence 

		if self.targeted:
			loss1 = clamp(other - real + c, min=0.)
		else:
			loss1 = clamp(real - other + c, min=0.)




		loss2 = (l2distsq).sum()
		loss1 = torch.sum(const * loss1)



		loss = loss1 + loss2
		return loss

	def _is_successful(self, output, label, is_logits, pred_labels=None,index=None):
		# determine success, see if confidence-adjusted logits give the right
		#   label

		if is_logits:
			output = output.detach().clone()
			if self.targeted:
				output[torch.arange(len(label)).long(),
					   label] -= self.confidence
			else:
				output[torch.arange(len(label)).long(),
					   label] += self.confidence
			con, pred = F.softmax(output, dim=1).max(1)
			evidence = output.logsumexp(dim=1)
		else:
			pred = pred_labels
			# print("pred:",pred)
			# print("INVAL_LABEL",INVALID_LABEL)
			if pred == INVALID_LABEL:
				# print("!!!!did ==")
				return pred.new_zeros(pred.shape).byte() 
			con = F.softmax(output, dim=0).max(0)[0]
			evidence = output.logsumexp(dim=0)

		if label.dim()!=0:
			return is_successful(pred, label, self.targeted)  * (self.dml_decision * self.dis_decision * self.ssim_decision)#needing modification for adaptive attacks
		else:
			# print("dml_decision_len",len(self.dml_decision))
			# print("ssim_len:",len(self.ssim_decision))
			return is_successful(pred, label, self.targeted)  * (self.dml_decision[index] * self.dis_decision[index] * self.ssim_decision[index])

	def _forward_and_update_delta(
			self, optimizer, x_atanh, delta, y_onehot, loss_coeffs):

		optimizer.zero_grad()
		adv = tanh_rescale(delta + x_atanh, self.clip_min, self.clip_max)# adv belongs to [0,1] is an image atanh belongs to -inf +linf

		adv_scale = adv*2 - 1 # generator wants [-1, 1]

		transimgs_rescale = tanh_rescale(x_atanh, self.clip_min, self.clip_max)

		output = self.predict(normalize(adv))
		l2distsq = calc_l2distsq(adv, transimgs_rescale) #caculate l2 distance 
		loss_cw = self._loss_fn(output, y_onehot, l2distsq, loss_coeffs)
		
		latent_i = self.encoder(adv_scale)
		z_mean, z_log_var, z = self.vae(latent_i)
		y_to_contranet = torch.argmax(y_onehot, dim=1)
		fake = self.gen(z, y_to_contranet)
		
		feat_emb, _ = self.orimodel(adv_scale)#needs [-1,1]
		feat_pos, _ = self.genmodel(fake)
		pos_pair = torch.cat([feat_emb, feat_pos], dim=1)
		y_mlp_pos = torch.ones(len(adv_scale)).long().to('cuda')
		pos_pred = self.mlpModel((pos_pair, y_to_contranet))

		dis_out_fake = self.discriminator(fake, y_to_contranet)
	
		D_loss = 0 - torch.mean(dis_out_fake)*1.0
		loss_mlp = mlploss(pos_pred, y_mlp_pos)
		loss_img_l2 = torch.abs((fake - adv_scale)**2).mean()
		loss_img_ssim = -torch.log(ssim_size_average(fake, adv_scale).mean() + 1e-15)
		# TODO: make an option for the adaptive terms taking into account
		adaptive_loss = self.adaptive_loss




		
		if adaptive_loss == "all_lambda":
			adaptive_cw_loss = loss_cw + (D_loss + loss_mlp +  loss_img_ssim)*torch.mean(loss_coeffs)
		elif adaptive_loss == "dis_dml":
			adaptive_cw_loss = loss_cw + loss_mlp + D_loss
		elif adaptive_loss == "ssim_dis_dml":
			adaptive_cw_loss = loss_img_ssim + loss_cw + loss_mlp +D_loss
		elif adaptive_loss == "ssim_dis":
			adaptive_cw_loss =  loss_img_ssim + loss_cw + D_loss
		elif adaptive_loss == "ssim_dml":
			adaptive_cw_loss = loss_img_ssim + loss_cw + loss_mlp
		elif adaptive_loss == "dml":
			adaptive_cw_loss = loss_cw + loss_mlp
		elif adaptive_loss == "dis":
			adaptive_cw_loss = loss_cw + D_loss
		elif adaptive_loss == "ssim":
			adaptive_cw_loss = loss_img_ssim + loss_cw
		elif adaptive_loss =="all":
			adaptive_cw_loss = loss_cw + (D_loss + loss_mlp +  loss_img_ssim)
		else:
			raise Exception("adaptive loss not implemented!!!")





		# used to judge attack successful or not
		dis_threshold = -1.334
		dml_threshold = 0.0107
		ssim_threshold = 0.2365
		pred_y = (pos_pred[:, 1] > dml_threshold).long()
		
		self.dml_decision = pred_y == 1
		self.dis_decision =  (dis_out_fake > dis_threshold)*1 == 1
		self.ssim_decision =  (ssim_size_average(fake, adv_scale) > ssim_threshold)
		
		
		adaptive_cw_loss.backward()
		optimizer.step()

		return adaptive_cw_loss.item(), l2distsq.data, output.data, adv.data


	def _get_arctanh_x(self, x):
		result = clamp((x - self.clip_min) / (self.clip_max - self.clip_min),
					   min=0., max=1.) * 2 - 1#scale to [-1,1]
		return torch_arctanh(result * ONE_MINUS_EPS)

	def _update_if_smaller_dist_succeed(
			self, adv_img, labs, output, l2distsq, batch_size,
			cur_l2distsqs, cur_labels,
			final_l2distsqs, final_labels, final_advs,
			cur_output):

		target_label = labs
		output_logits = output
		_, output_label = torch.max(output_logits, 1)

		mask = (l2distsq < cur_l2distsqs) & self._is_successful(
			output_logits, target_label, True)

		cur_l2distsqs[mask] = l2distsq[mask]  # redundant
		cur_labels[mask] = output_label[mask]
		cur_output[mask,:] = output_logits[mask,:]

		mask = (l2distsq < final_l2distsqs) & self._is_successful(
			output_logits, target_label, True)
		final_l2distsqs[mask] = l2distsq[mask]
		final_labels[mask] = output_label[mask]
		final_advs[mask] = adv_img[mask] 



	def _update_loss_coeffs(
		self, labs, cur_labels, batch_size, loss_coeffs,
		coeff_upper_bound, coeff_lower_bound,
		cur_output
	):
		# TODO: remove for loop, not significant, since only called during each binary search step
		for ii in range(batch_size):
			cur_labels[ii] = int(cur_labels[ii])
			if self._is_successful(cur_output[ii], labs[ii], False, pred_labels=cur_labels[ii],index=ii):# False refer to is_logits
				coeff_upper_bound[ii] = min(coeff_upper_bound[ii], loss_coeffs[ii])

				if coeff_upper_bound[ii] < UPPER_CHECK:
					loss_coeffs[ii] = (coeff_lower_bound[ii] + coeff_upper_bound[ii])/2
			else:# Step1: change the upper/lower bound according to attack results. once attack is successful, decreasing the upper bound, else increase the lower bound
				coeff_lower_bound[ii] = max(
					coeff_lower_bound[ii], loss_coeffs[ii])
				if coeff_upper_bound[ii] < UPPER_CHECK:
					loss_coeffs[ii] = (
						coeff_lower_bound[ii] + coeff_upper_bound[ii]
					)/2


				else:
					loss_coeffs[ii] *= 10

	def perturb(self, x, y=None):
		x, y = self._verify_and_process_inputs(x, y)
		# Initialization
		if y is None:
			y = self._get_predicted_label(x)
		x = replicate_input(x)
		batch_size = len(x)
		coeff_lower_bound = x.new_zeros(batch_size)
		coeff_upper_bound = x.new_ones(batch_size)* CARLINI_COEFF_UPPER
		# print("changed")
		loss_coeffs = torch.ones_like(y).float() * self.initial_const
		final_l2distsqs = [CARLINI_COEFF_UPPER] * batch_size
		final_labels = [INVALID_LABEL] * batch_size
		final_advs = x
		x_atanh = self._get_arctanh_x(x)
		y_onehot = to_one_hot(y, self.num_classes).float()

		final_l2distsqs = torch.FloatTensor(final_l2distsqs).to(x.device)
		final_labels = torch.LongTensor(final_labels).to(x.device)
		misc_output = torch.zeros(x.size()[0], self.num_classes).float().cuda()
		# Start binary search
		for outer_step in range(self.binary_search_steps): #binary_search_steps == 9 
			#import ipdb; ipdb.set_trace()
			delta = nn.Parameter(torch.zeros_like(x))

			optimizer = optim.Adam([delta], lr=self.learning_rate)#learning_rate =0.01
			cur_l2distsqs = [CARLINI_L2DIST_UPPER] * batch_size
			cur_labels = [INVALID_LABEL] * batch_size
			cur_l2distsqs = torch.FloatTensor(cur_l2distsqs).to(x.device)
			cur_labels = torch.LongTensor(cur_labels).to(x.device)
			prevloss = PREV_LOSS_INIT

			# record current output
			cur_output = torch.zeros(x.size()[0], self.num_classes).float().cuda()

			if (self.repeat and outer_step==(self.binary_search_steps - 1)):
				loss_coeffs = coeff_upper_bound
			for ii in range(self.max_iterations):
				loss, l2distsq, output, adv_img = \
					self._forward_and_update_delta(optimizer, x_atanh, delta, y_onehot, loss_coeffs)
				if self.abort_early:
					if ii %(self.max_iterations//NUM_CHECKS or 1) == 0:
						if loss > prevloss * ONE_MINUS_EPS:
							break 
						pervloss = loss

				self._update_if_smaller_dist_succeed(
					adv_img, y,output, l2distsq, batch_size,
					cur_l2distsqs, cur_labels,
					final_l2distsqs, final_labels, final_advs, cur_output
				)

			self._update_loss_coeffs(
				y, cur_labels, batch_size,
				loss_coeffs, coeff_upper_bound, coeff_lower_bound,
				cur_output
			)

		return final_advs#, self._is_successful(misc_output, y, False, final_labels) #adding a decision term 





