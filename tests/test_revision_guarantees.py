"""Adversarial checks for the repaired guarantees and information boundaries."""
import math
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.stats import norm
from sklearn.covariance import OAS, empirical_covariance
from sklearn.neighbors import NearestNeighbors
from src.graphcal.activation import activation_decision
from src.graphcal.benchmark_data import load_revision_dataset
from src.graphcal.preprocessing import TrainingEncoder
from src.graphcal.augmenter import (
    GraphCalAugmenter, k_nearest_indices, scaled_identity_ledoit_wolf, voronoi_indices,
)
from src.graphcal.protocol import PAIRED_SETTINGS, comparison_family_size


class RevisionGuarantees(unittest.TestCase):
    def test_uniform_gaussian_residual_does_not_vanish(self):
        # W2 lower bound from Gaussian mass outside the uniform's support.
        lower=2*quad(lambda x:(x-1)**2*norm.pdf(x,scale=1/np.sqrt(3)),1,np.inf)[0]
        self.assertGreater(lower,0.001)

    def test_sparse_valid_points_can_fail_global_filter(self):
        dense=np.column_stack((np.cos(np.linspace(0,0.02,400)),np.sin(np.linspace(0,0.02,400))))
        sparse=np.column_stack((np.cos(np.linspace(2,2.3,10)),np.sin(np.linspace(2,2.3,10))))
        from sklearn.neighbors import NearestNeighbors
        real=np.vstack((dense,sparse));k=5
        score=NearestNeighbors(n_neighbors=k+1).fit(real).kneighbors(real)[0][:,1:].mean(1)
        self.assertGreater(float(np.median(score[-10:])),2*float(np.median(score)))

    def test_gate_uses_bounded_contrasts_and_multiplicity(self):
        y=np.zeros(1000);base=np.ones(1000);candidate=np.zeros(1000)
        a=activation_decision(y,base,candidate,loss_cap=1,family_size=1)
        b=activation_decision(y,base,candidate,loss_cap=1,family_size=20)
        self.assertTrue(a.activated)
        self.assertGreater(b.hoeffding_margin,a.hoeffding_margin)
        self.assertFalse(activation_decision(y,base,base,loss_cap=1,family_size=20).activated)

    def test_capped_certificate_is_not_raw_mae_certificate(self):
        y=np.zeros(1000);base=np.full(1000,.5);candidate=np.zeros(1000);candidate[-1]=1000
        self.assertTrue(activation_decision(y,base,candidate,loss_cap=1,family_size=1).activated)
        self.assertGreater(np.abs(candidate).mean(),np.abs(base).mean())

    def test_loo_encoding_excludes_own_category_and_global_label(self):
        frame=pd.DataFrame({'category':['a','a','b','c'],'x':[1.,2.,3.,4.]})
        y=np.array([1.,2.,4.,8.]);changed=y.copy();changed[0]=900
        a=TrainingEncoder(('category',)).fit_transform(frame,y)
        b=TrainingEncoder(('category',)).fit_transform(frame,changed)
        self.assertEqual(a[0,0],b[0,0])

    def test_augmenter_duplicate_rows_and_small_inputs(self):
        x=np.array([[0.,0.],[0.,0.],[1.,1.],[2.,2.]])
        result=GraphCalAugmenter(k=3,random_state=7).fit_transform(x,x[:,0])
        self.assertTrue(np.isfinite(result.X_aug).all())
        with self.assertRaises(ValueError):GraphCalAugmenter().fit_transform(x[:2],x[:2,0])

    def test_no_sparse_path_keeps_original_sample_and_fitted_state(self):
        x=np.ones((8,2));y=np.ones(8)
        aug=GraphCalAugmenter(random_state=0)
        result=aug.fit_transform(x,y)
        self.assertEqual(result.diagnostics['num_sparse_nodes'],0)
        np.testing.assert_array_equal(result.X_aug,x)
        np.testing.assert_array_equal(result.y_aug,y)
        self.assertEqual(aug.sparse_indices_.size,0)
        self.assertEqual(aug.reference_z_.shape,(8,3))
        self.assertTrue(np.isfinite(aug.reference_z_).all())
        encoded=aug.encode_joint(x,y)
        np.testing.assert_allclose(encoded,aug.reference_z_)
        assigned=voronoi_indices(encoded,aug.reference_z_)
        self.assertEqual(assigned.shape,(8,))
        self.assertFalse(np.isin(assigned,aug.sparse_indices_).any())

    def test_ledoit_wolf_target_is_scaled_identity(self):
        rng=np.random.default_rng(0)
        x=rng.normal(size=(40,4));x[:,1]+=2*x[:,0]
        cov=scaled_identity_ledoit_wolf(x)
        S=empirical_covariance(x)
        scale=float(np.trace(S)/S.shape[0])
        # Recover lambda from a diagonal entry that the target moves.
        diagonal_gap=np.diag(S)-scale
        moved=np.flatnonzero(np.abs(diagonal_gap)>1e-8)
        lam=float((np.diag(cov)[moved[0]]-np.diag(S)[moved[0]])/(scale-np.diag(S)[moved[0]]))
        expected=(1-lam)*S
        expected.flat[::S.shape[0]+1]+=lam*scale
        np.testing.assert_allclose(cov,expected,atol=1e-10)
        diagonal_target=(1-lam)*S+lam*np.diag(np.diag(S))
        self.assertFalse(np.allclose(cov,diagonal_target))
        self.assertFalse(np.allclose(cov,OAS().fit(x).covariance_))
        off=cov-np.diag(np.diag(cov))
        np.testing.assert_allclose(off,(1-lam)*(S-np.diag(np.diag(S))),atol=1e-10)

    def test_neighborhood_rule_is_the_same_in_low_and_high_dimension(self):
        rng=np.random.default_rng(1)
        low=GraphCalAugmenter(random_state=1).fit_transform(rng.normal(size=(60,2)),rng.normal(size=60))
        self.assertEqual(low.diagnostics['joint_dimension'],3)
        self.assertEqual(low.diagnostics['required_neighborhood'],6)
        self.assertGreater(low.diagnostics['num_sparse_nodes'],0)
        self.assertEqual(low.diagnostics['anchors_below_2p'],0)
        self.assertLessEqual(low.diagnostics['max_hops_used'],4)
        high_x=rng.normal(size=(30,20));high_y=rng.normal(size=30)
        with patch('src.graphcal.augmenter.scaled_identity_ledoit_wolf', wraps=scaled_identity_ledoit_wolf) as spy:
            high=GraphCalAugmenter(random_state=1).fit_transform(high_x,high_y)
        self.assertEqual(high.diagnostics['joint_dimension'],21)
        self.assertEqual(high.diagnostics['required_neighborhood'],42)
        self.assertGreater(high.diagnostics['anchors_below_2p'],0)
        self.assertEqual(high.diagnostics['max_hops_used'],4)
        self.assertGreater(spy.call_count,0)
        self.assertTrue(np.isfinite(high.X_aug).all())

    def test_distance_screen_runs_before_residual_and_append(self):
        rng=np.random.default_rng(2)
        x=rng.normal(size=(40,3));y=x.sum(1)+rng.normal(size=40)
        aug=GraphCalAugmenter(kappa=3,gamma=2,random_state=2)
        result=aug.fit_transform(x,y)
        calls=[]
        real=aug.residual_accepts
        def wrapped(cand,idx):
            calls.append(1)
            return real(cand,idx)
        aug.residual_accepts=wrapped
        self.assertFalse(aug.passes_screens(np.full(aug.reference_z_.shape[1], 80.)))
        self.assertEqual(calls,[])
        aug.passes_screens(aug.reference_z_[0])
        self.assertEqual(calls,[1])
        extra=len(result.y_aug)-len(y)
        self.assertGreater(extra,0)
        syn=aug.encode_joint(result.X_aug[len(y):],result.y_aug[len(y):])
        for cand in syn:
            distances=np.linalg.norm(aug.reference_z_-cand,axis=1)
            score=float(distances[k_nearest_indices(distances,aug.k_)].mean())
            self.assertLessEqual(score,aug.threshold_+1e-6)

    def test_voronoi_tie_uses_smallest_index_when_sklearn_does_not(self):
        rng=np.random.default_rng(0)
        base=rng.normal(size=(40,5))
        ref=np.vstack((base,base))
        query=base[:8].copy()
        idx=voronoi_indices(query,ref)
        self.assertTrue(np.all(idx<40))
        sklearn_idx=NearestNeighbors(n_neighbors=1).fit(ref).kneighbors(query,return_distance=False).ravel()
        self.assertGreater(int(sklearn_idx[4]),4)
        self.assertEqual(int(idx[4]),4)
        circle=np.array([[1.,0.],[0.,1.],[-1.,0.],[0.,-1.]])
        self.assertEqual(int(voronoi_indices(np.array([[0.,0.]]),circle)[0]),0)
        tied=np.ones((5000,1));tied[0,0]=-1;tied[4500,0]=1
        self.assertEqual(int(voronoi_indices(np.array([[0.]]),tied,block=1024)[0]),0)

    def test_activation_margin_and_family_are_within_one_comparison(self):
        m,K,M,eta=50,7,2.5,0.05
        decision=activation_decision(np.zeros(m),np.ones(m),np.zeros(m),loss_cap=M,family_size=K,failure_probability=eta)
        self.assertAlmostEqual(decision.hoeffding_margin,M*math.sqrt(2*math.log(K/eta)/m))
        methods=('vanilla','smoter','smogn','lds','balanced_mse','conr','prime','cems','graphcal')
        self.assertEqual(comparison_family_size(methods),8)
        self.assertNotEqual(comparison_family_size(methods),8*9*5)
        self.assertEqual([(s['learning_rate'],s['kappa']) for s in PAIRED_SETTINGS],[(1e-3,2.0),(3e-3,1.0)])

    def test_energy_cache_has_eight_inputs_and_no_y2(self):
        frame,y,meta=load_revision_dataset('energy')
        self.assertEqual(frame.shape[1],8)
        self.assertEqual(len(y),768)
        self.assertNotIn('y2',frame.columns)
        self.assertEqual(meta['categorical'],['V6','V8'])


if __name__=='__main__':unittest.main()
