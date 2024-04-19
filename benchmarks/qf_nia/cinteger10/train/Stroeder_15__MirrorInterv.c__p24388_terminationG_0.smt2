(set-info :smt-lib-version 2.6)
(set-logic QF_NIA)
(set-info :source |
Generated by: Cristina Borralleras, Daniel Larraz, Albert Oliveras, Enric Rodriguez-Carbonell, Albert Rubio
Generated on: 2017-04-27
Generator: VeryMax
Application: Termination proving
Target solver: barcelogic
|)
(set-info :license "https://creativecommons.org/licenses/by/4.0/")
(set-info :category "industrial")
(set-info :status unknown)
(declare-fun Nl2main_i1 () Int)
(declare-fun Nl2main_range1 () Int)
(declare-fun Nl2main_i2 () Int)
(declare-fun Nl2main_range2 () Int)
(declare-fun lam0n0 () Int)
(declare-fun lam0n1 () Int)
(declare-fun lam0n2 () Int)
(declare-fun lam0n3 () Int)
(declare-fun lam0n4 () Int)
(declare-fun Nl2CT1 () Int)
(declare-fun Nl2CT2 () Int)
(declare-fun lam1n0 () Int)
(declare-fun lam1n1 () Int)
(declare-fun lam1n2 () Int)
(declare-fun lam1n3 () Int)
(declare-fun lam1n4 () Int)
(declare-fun lam2n0 () Int)
(declare-fun lam2n1 () Int)
(declare-fun lam2n2 () Int)
(declare-fun lam2n3 () Int)
(declare-fun lam2n4 () Int)
(declare-fun lam3n0 () Int)
(declare-fun lam3n1 () Int)
(declare-fun lam3n3 () Int)
(declare-fun lam3n2 () Int)
(declare-fun lam4n0 () Int)
(declare-fun lam4n1 () Int)
(declare-fun lam4n3 () Int)
(declare-fun lam4n2 () Int)
(declare-fun main_i () Int)
(declare-fun main_range () Int)
(declare-fun undef2 () Int)
(declare-fun lam8n0 () Int)
(declare-fun lam8n1 () Int)
(declare-fun lam8n2 () Int)
(declare-fun lam8n3 () Int)
(declare-fun lam8n4 () Int)
(declare-fun RFN1_CT () Int)
(declare-fun RFN1_main_i () Int)
(declare-fun RFN1_main_range () Int)
(declare-fun lam6n0 () Int)
(declare-fun lam6n1 () Int)
(declare-fun lam6n2 () Int)
(declare-fun lam6n3 () Int)
(declare-fun lam6n4 () Int)
(declare-fun lam7n0 () Int)
(declare-fun lam7n1 () Int)
(declare-fun lam7n2 () Int)
(declare-fun lam7n3 () Int)
(declare-fun lam7n4 () Int)
(declare-fun lam5n0 () Int)
(declare-fun lam5n1 () Int)
(declare-fun lam5n2 () Int)
(declare-fun lam5n3 () Int)
(declare-fun lam5n4 () Int)
(assert ( and ( <= ( - 1 ) Nl2main_i1 ) ( <= Nl2main_i1 1 ) ( <= ( - 1 ) Nl2main_range1 ) ( <= Nl2main_range1 1 ) ( <= ( - 1 ) Nl2main_i2 ) ( <= Nl2main_i2 1 ) ( <= ( - 1 ) Nl2main_range2 ) ( <= Nl2main_range2 1 ) ))
(assert ( or ( and ( >= lam0n0 0 ) ( >= lam0n1 0 ) ( >= lam0n2 0 ) ( >= lam0n3 0 ) ( >= lam0n4 0 ) ( > ( + ( * 5 lam0n0 ) ( * 5 lam0n1 ) ( * 1 lam0n2 ) ( * Nl2CT1 lam0n3 ) ( * Nl2CT2 lam0n4 ) ( - 1 ) ) 0 ) ( = ( + ( * 1 lam0n0 ) ( * ( - 1 ) lam0n1 ) ( * 1 lam0n2 ) ( * Nl2main_i1 lam0n3 ) ( * Nl2main_i2 lam0n4 ) ) 0 ) ( = ( + ( * ( - 1 ) lam0n0 ) ( * ( - 1 ) lam0n1 ) ( * Nl2main_range1 lam0n3 ) ( * Nl2main_range2 lam0n4 ) ) 0 ) ) ( and ( >= lam1n0 0 ) ( >= lam1n1 0 ) ( >= lam1n2 0 ) ( >= lam1n3 0 ) ( >= lam1n4 0 ) ( > ( + ( * 5 lam1n0 ) ( * 5 lam1n1 ) ( * 1 lam1n2 ) ( * Nl2CT1 lam1n3 ) ( * Nl2CT2 lam1n4 ) ( - 1 ( + ( + Nl2CT1 ( * Nl2main_i1 ( - 1 ) ) ) ( * Nl2main_range1 1 ) ) ) ) 0 ) ( = ( + ( * 1 lam1n0 ) ( * ( - 1 ) lam1n1 ) ( * 1 lam1n2 ) ( * Nl2main_i1 lam1n3 ) ( * Nl2main_i2 lam1n4 ) ( - ( + 0 ( * Nl2main_i1 1 ) ) ) ) 0 ) ( = ( + ( * ( - 1 ) lam1n0 ) ( * ( - 1 ) lam1n1 ) ( * Nl2main_range1 lam1n3 ) ( * Nl2main_range2 lam1n4 ) ( - ( + 0 ( * Nl2main_range1 1 ) ) ) ) 0 ) ) ))
(assert ( or ( and ( >= lam0n0 0 ) ( >= lam0n1 0 ) ( >= lam0n2 0 ) ( >= lam0n3 0 ) ( >= lam0n4 0 ) ( > ( + ( * 5 lam0n0 ) ( * 5 lam0n1 ) ( * 1 lam0n2 ) ( * Nl2CT1 lam0n3 ) ( * Nl2CT2 lam0n4 ) ( - 1 ) ) 0 ) ( = ( + ( * 1 lam0n0 ) ( * ( - 1 ) lam0n1 ) ( * 1 lam0n2 ) ( * Nl2main_i1 lam0n3 ) ( * Nl2main_i2 lam0n4 ) ) 0 ) ( = ( + ( * ( - 1 ) lam0n0 ) ( * ( - 1 ) lam0n1 ) ( * Nl2main_range1 lam0n3 ) ( * Nl2main_range2 lam0n4 ) ) 0 ) ) ( and ( >= lam2n0 0 ) ( >= lam2n1 0 ) ( >= lam2n2 0 ) ( >= lam2n3 0 ) ( >= lam2n4 0 ) ( > ( + ( * 5 lam2n0 ) ( * 5 lam2n1 ) ( * 1 lam2n2 ) ( * Nl2CT1 lam2n3 ) ( * Nl2CT2 lam2n4 ) ( - 1 ( + ( + Nl2CT2 ( * Nl2main_i2 ( - 1 ) ) ) ( * Nl2main_range2 1 ) ) ) ) 0 ) ( = ( + ( * 1 lam2n0 ) ( * ( - 1 ) lam2n1 ) ( * 1 lam2n2 ) ( * Nl2main_i1 lam2n3 ) ( * Nl2main_i2 lam2n4 ) ( - ( + 0 ( * Nl2main_i2 1 ) ) ) ) 0 ) ( = ( + ( * ( - 1 ) lam2n0 ) ( * ( - 1 ) lam2n1 ) ( * Nl2main_range1 lam2n3 ) ( * Nl2main_range2 lam2n4 ) ( - ( + 0 ( * Nl2main_range2 1 ) ) ) ) 0 ) ) ))
(assert ( and ( not ( <= ( + 1 ( * ( - 1 ) main_i ) main_range ) 0 ) ) ( not ( <= ( + 1 main_i main_range ) 0 ) ) ( = ( + main_i ( * ( - 1 ) undef2 ) ) 0 ) ( = ( + ( - 20 ) main_range ) 0 ) ( <= ( + Nl2CT1 ( * ( + 0 Nl2main_i1 ) main_i ) ( * ( + 0 Nl2main_range1 ) main_range ) ) 0 ) ( <= ( + Nl2CT2 ( * ( + 0 Nl2main_i2 ) main_i ) ( * ( + 0 Nl2main_range2 ) main_range ) ) 0 ) ( <= ( + 1 main_i ) 0 ) ( <= ( + 5 ( * ( - 1 ) main_i ) ( * ( - 1 ) main_range ) ) 0 ) ))
(assert ( or ( and ( and ( >= lam8n0 0 ) ( >= lam8n1 0 ) ( >= lam8n2 0 ) ( >= lam8n3 0 ) ( >= lam8n4 0 ) ( > ( + ( * 5 lam8n0 ) ( * 5 lam8n1 ) ( * 50001 lam8n2 ) ( * Nl2CT1 lam8n3 ) ( * Nl2CT2 lam8n4 ) ( - 1 ( - ( + ( + RFN1_CT ( * RFN1_main_i ( - 1 ) ) ) ( * RFN1_main_range 1 ) ) RFN1_CT ) ) ) 0 ) ( = ( + ( * 1 lam8n0 ) ( * ( - 1 ) lam8n1 ) ( * 1 lam8n2 ) ( * Nl2main_i1 lam8n3 ) ( * Nl2main_i2 lam8n4 ) ( - ( - ( + 0 ( * RFN1_main_i 1 ) ) RFN1_main_i ) ) ) 0 ) ( = ( + ( * ( - 1 ) lam8n0 ) ( * ( - 1 ) lam8n1 ) ( * Nl2main_range1 lam8n3 ) ( * Nl2main_range2 lam8n4 ) ( - ( - ( + 0 ( * RFN1_main_range 1 ) ) RFN1_main_range ) ) ) 0 ) ) ( and ( and ( >= lam6n0 0 ) ( >= lam6n1 0 ) ( >= lam6n2 0 ) ( >= lam6n3 0 ) ( >= lam6n4 0 ) ( > ( + ( * 5 lam6n0 ) ( * 5 lam6n1 ) ( * 50001 lam6n2 ) ( * Nl2CT1 lam6n3 ) ( * Nl2CT2 lam6n4 ) ( - 1 ( - RFN1_CT ) ) ) 0 ) ( = ( + ( * 1 lam6n0 ) ( * ( - 1 ) lam6n1 ) ( * 1 lam6n2 ) ( * Nl2main_i1 lam6n3 ) ( * Nl2main_i2 lam6n4 ) ( - ( - RFN1_main_i ) ) ) 0 ) ( = ( + ( * ( - 1 ) lam6n0 ) ( * ( - 1 ) lam6n1 ) ( * Nl2main_range1 lam6n3 ) ( * Nl2main_range2 lam6n4 ) ( - ( - RFN1_main_range ) ) ) 0 ) ) ( and ( >= lam7n0 0 ) ( >= lam7n1 0 ) ( >= lam7n2 0 ) ( >= lam7n3 0 ) ( >= lam7n4 0 ) ( > ( + ( * 5 lam7n0 ) ( * 5 lam7n1 ) ( * 50001 lam7n2 ) ( * Nl2CT1 lam7n3 ) ( * Nl2CT2 lam7n4 ) ( - 1 ( + ( - ( + ( + RFN1_CT ( * RFN1_main_i ( - 1 ) ) ) ( * RFN1_main_range 1 ) ) RFN1_CT ) 1 ) ) ) 0 ) ( = ( + ( * 1 lam7n0 ) ( * ( - 1 ) lam7n1 ) ( * 1 lam7n2 ) ( * Nl2main_i1 lam7n3 ) ( * Nl2main_i2 lam7n4 ) ( - ( - ( + 0 ( * RFN1_main_i 1 ) ) RFN1_main_i ) ) ) 0 ) ( = ( + ( * ( - 1 ) lam7n0 ) ( * ( - 1 ) lam7n1 ) ( * Nl2main_range1 lam7n3 ) ( * Nl2main_range2 lam7n4 ) ( - ( - ( + 0 ( * RFN1_main_range 1 ) ) RFN1_main_range ) ) ) 0 ) ) ) ) ( and ( >= lam5n0 0 ) ( >= lam5n1 0 ) ( >= lam5n2 0 ) ( >= lam5n3 0 ) ( >= lam5n4 0 ) ( > ( + ( * 5 lam5n0 ) ( * 5 lam5n1 ) ( * 50001 lam5n2 ) ( * Nl2CT1 lam5n3 ) ( * Nl2CT2 lam5n4 ) ( - 1 ) ) 0 ) ( = ( + ( * 1 lam5n0 ) ( * ( - 1 ) lam5n1 ) ( * 1 lam5n2 ) ( * Nl2main_i1 lam5n3 ) ( * Nl2main_i2 lam5n4 ) ) 0 ) ( = ( + ( * ( - 1 ) lam5n0 ) ( * ( - 1 ) lam5n1 ) ( * Nl2main_range1 lam5n3 ) ( * Nl2main_range2 lam5n4 ) ) 0 ) ) ))
(check-sat)
(exit)
