from collections import defaultdict as dd, Counter
import csv,re
import numpy as np
from nltk import word_tokenize as wtok
from scipy.stats import hmean
from subprocess import call
import os

def read_parameters(fn):
	# gets the P(t|s) in a dictionary format
	current_word = None
	parameters = dd(lambda : dd(float))
	for l in open(fn).readlines():
		if l[0] == '#': continue
		elif l[0] != ' ':
			fields = re.split('\t', l)
			if len(fields) < 3: continue
			current_word = fields[0]
		else:
			fields = re.split(': ', l[2:])
			parameters[current_word][fields[0]] = float(fields[1])
	return parameters

def read_phrases(filename):
	# reads the list of phrases the model searches for
	lines = open(filename).readlines()
	phrases = set()
	for l in lines:
		phrases.add(tuple(re.split(' ', l.strip('\n'))))
	return phrases

def is_phrase(words, phrases, anchor = 'start'):
	# returns all maximal* spans in a sentence that contain a phrase from the list
	# * maximal meaning that there are no superstrings of that string found as phrases as well
	longest_phrase = np.max([len(f) for f in phrases])
	indices = {}
	for i in range(0,(len(words) if anchor != 'start' else 1)):
		for j in range(i+1, np.min([i+1+longest_phrase, len(words)])):
			if tuple(words[i:j]) in phrases:
				try:
					if indices[i] < j: indices[i] = j
				except KeyError:
					indices[i] = j
	return indices

def process_corpus(l_s = 'nl', l_t = 'tr', dirname = '', 
	               shared_fn = 'OpenSubtitles2016', 
	               phrases_fn = 'phrases.txt',
	               id_constraint = None):
	# extracts all translations of found phrases
	phrases = read_phrases(phrases_fn)
	l_a, l_b, para_a, para_b = (l_s,l_t,1,2) if l_s < l_t else (l_t,l_s,2,1)
	#
	fp_s = '%s/%s_%s/out/hmm/%d.params.txt' % (dirname, l_a, l_b, para_a)
	fp_t = '%s/%s_%s/out/hmm/%d.params.txt' % (dirname, l_a, l_b, para_b)
	p_st, p_ts = read_parameters(fp_s), read_parameters(fp_t)
	# read parameters from Liang's model
	fn = '%s/%s_%s/%s.%s-%s' % (dirname, l_a, l_b, shared_fn, l_a, l_b)
	f_s, f_t, f_i = open('%s.%s' % (fn,l_s)), open('%s.%s' % (fn,l_t)), open('%s.ids' % (fn))
	#all_translate = dd(lambda : Counter())
	ctr = 0	
	out = open('output.csv', 'a')
	li_s, li_t, ids = f_s.readline(), f_t.readline(), re.split('\t', f_i.readline().strip('\n'))
	while ids != ['']:
		ctr += 1
		if ctr % 100000 == 0: print(l_t, ctr)
		if id_constraint == [] or ids[para_a-1] in id_constraint:
			s_ix = '%s_%s' % (ids[para_a-1],ids[para_a+1]) # para_a = 1 if < else 2
			t_ix = '%s_%s' % (ids[para_b-1],ids[para_b+1]) # para_b = 2 if < else 1
			li_s, li_t = re.sub('^\s?-', '- ', li_s.strip('\n').lower()), re.sub('^\s?-', '- ', li_t.strip('\n').lower())
			#if ctr > 2000: break
			u_s, u_t = wtok(li_s), wtok(li_t)
			indices = is_phrase(u_s, phrases, 'start')
			#if len(indices) <= 1: indices = {}
			for start_s, end_s in indices.items():
				str_s, str_t = align(p_st, p_ts, u_s, u_t, start_s, end_s)
				#print('%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' % (l_s, l_t, s_ix, t_ix, str_s, str_t, ' '.join(u_s), ' '.join(u_t)))
				out.write('%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' % (l_s, l_t, s_ix, t_ix, str_s, str_t, ' '.join(u_s), ' '.join(u_t)))
				#all_translate[str_s][str_t] += 1
		li_s, li_t, ids = f_s.readline(), f_t.readline(),  re.split('\t', f_i.readline().strip('\n'))
	out.close()
	return

def align(p_st, p_ts, u_s, u_t, start_s, end_s):
	# aligns a source phrase to a target phrase; see technical report for description
	np.set_printoptions(precision = 2, suppress = True, linewidth = 200)
	wcount_s, wcount_t = Counter(u_s), Counter(u_t)
	w_factor_s, w_factor_t = np.array([wcount_s[w] for w in u_s]), np.array([wcount_t[w] for w in u_t])
	# used for double words
	aligns_s = np.array([np.array([p_st[w_s][w_t] + 0.000001 for w_t in u_t]) for w_s in u_s])
	aligns_s = ((aligns_s / aligns_s.sum(0)).T * w_factor_s).T
	aligns_t = np.array([np.array([p_ts[w_t][w_s] + 0.000001 for w_s in u_s]) for w_t in u_t])
	aligns_t = ((aligns_t / aligns_t.sum(0)).T * w_factor_t).T
	#
	breaks = np.zeros((len(u_t),len(u_t)))
	excl_range_s = list(range(start_s)) + list(range(end_s,len(u_s)))
	for i in range(len(u_t)-1):
		for j in range(i,len(u_t)):
			excl_range_t = list(range(i)) + list(range(j,len(u_t)))
			excl_t = aligns_s[excl_range_s].T[excl_range_t].T
			incl_t = aligns_s[start_s:end_s, i:j]
			excl_s = aligns_t.T[excl_range_s].T[excl_range_t].T
			incl_s = aligns_t.T[start_s:end_s, i:j]

			score = np.mean( [ m.max(i).mean() if np.min(m.shape) > 0 else 0 for i in [0,1] 
							   for m in [incl_s, excl_s, incl_t, excl_t ] ] )
			breaks[i,j] = score
	start_t, end_t = np.unravel_index(breaks.argmax(), breaks.shape)
	s_str, t_str = ' '.join(u_s[start_s:end_s]), ' '.join(u_t[start_t:end_t])
	return s_str, t_str

###
# SAMPLING PART OF THE CORPUS (FOR V LARGE CORPORA)
###

def sample_train(l_s, l_t, wd , n_train):
	l_a, l_b = (l_s, l_t) if l_s < l_t else (l_t, l_s)
	folder1, folder2 = '%s_%s' % (l_a,l_b), '%s-%s' % (l_a,l_b)
	if os.path.isdir('%s/%s/sample' % (wd,folder1)): return
	call(['mkdir', '%s/%s/sample' % (wd,folder1)])
	fl1 = open('%s/%s/OpenSubtitles2016.%s.%s' % (wd, folder1,folder2,l_a))
	fl2 = open('%s/%s/OpenSubtitles2016.%s.%s' % (wd, folder1,folder2,l_b))
	outl1 = open('%s/%s/sample/texts.%s' % (wd, folder1, l_a), 'w')
	outl2 = open('%s/%s/sample/texts.%s' % (wd, folder1, l_b), 'w')
	ctr = 0
	line1, line2 = fl1.readline().strip('\n'), fl2.readline().strip('\n')
	while line1 != '' and line2 != '' and ctr < n_train:
		if ctr % 10000 == 0: print(l_t, ctr)
		u1t = ' '.join(wtok(re.sub('^\s?-', '- ', line1)))
		u2t = ' '.join(wtok(re.sub('^\s?-', '- ', line2)))
		outl1.write('%s\n' % u1t)
		outl2.write('%s\n' % u2t)
		ctr += 1
		line1, line2 = fl1.readline().strip('\n'), fl2.readline().strip('\n')
	outl1.close()
	outl2.close()

####
# PREPS AND SHELLS FOR LIANG MODEL
####

def rewrite_conf_file(l_s, languages, dirname):
	conf = { fields[0] : fields[1]
		     for fields in 
		     [re.split('\t', line.strip('\n')) 
		      for line in open('sample.conf').readlines()]
		      if len(fields) == 2 }
	for l_t in (languages):
		l_a, l_b = (l_s, l_t) if l_s < l_t else (l_t, l_s)
		folder = '%s_%s' % (l_a,l_b)
		conf['train'] = '%s/%s_%s/sample' % (dirname, l_a, l_b)
		conf['enExt'] = l_s	
		conf['frExt'] = l_t
		with open('%s/%s_%s/config.conf' % (dirname, l_a, l_b), 'w') as fh:
			for k,v in conf.items():
				fh.write('%s\t%s\n' % (k,v))

def run_liang_model(l_s, languages, dirname):
	# requires you to run this script from the same folder as where Liang's crossTrain executable is
	for l_t in languages:
		l_a, l_b = (l_s, l_t) if l_s < l_t else (l_t, l_s)
		conf = '%s/%s_%s/config.conf' % (dirname, l_a, l_b)
		out = '%s/%s_%s/out' % (dirname, l_a, l_b)
		if os.path.isdir(out): continue
		else: call(['./crossTrain', conf, out])

def main( languages = ['ko', 'tr', 'en', 'sv', 'id', 'ro', 'vi', 'pl', 'ca', 'lt', 'sr', 'et', 
					  'hu', 'hi', 'el', 'ka', 'ml', 'fa', 'ar', 'he', 'ja'],
		  source = 'nl',
		  dirname = '/Users/barendbeekhuizen/opus_sample_20'
		  id_constraint = None):
	rewrite_conf_file(source, languages, dirname)
	for l_t in languages: sample_train(source, l_t, dirname, 200000)
	run_liang_model(source, languages, dirname)
	for i,l_t in enumerate(languages):
		print(i,l_t)
		process_corpus(l_s = source, l_t = l_t, dirname = dirname, id_constraint = id_constraint)
	return

###
# OTHER FILMS
###

def shared_films(l_s, languages):
	film_freq = dd(int)
	l_ctr = 0
	for l_t in languages:
		l_ctr += 1
		para_a, l_a, l_b = (1,l_s, l_t) if l_s < l_t else (2,l_t, l_s)
		with open('/Users/barendbeekhuizen/opus_sample_20/%s_%s/OpenSubtitles2016.%s-%s.ids' % (l_a, l_b, l_a, l_b)) as fh:
			#call(['wc', '-l', '/Users/barendbeekhuizen/opus_sample_20/%s_%s/OpenSubtitles2016.%s-%s.ids' % (l_a, l_b, l_a, l_b)])
			line = re.split('\t', fh.readline().strip('\n'))
			prev_film = None
			ctr = 0
			while line != ['']:
				ctr += 1
				if ctr % 250000 == 0: print(ctr) 
				film, ix = line[para_a-1], line[para_a+1]
				if prev_film != film: 
					film_freq[film] += 1
					prev_film = film
				line = re.split('\t', fh.readline().strip('\n'))
			print('n films shared', len([k for k,v in film_freq.items() if v == l_ctr]))
	return film_freq

