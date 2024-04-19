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
(set-info :status sat)
(declare-fun Nl2main_x1 () Int)
(declare-fun Nl2main_y1 () Int)
(declare-fun Nl2main_z1 () Int)
(declare-fun Nl2main_x2 () Int)
(declare-fun Nl2main_y2 () Int)
(declare-fun Nl2main_z2 () Int)
(declare-fun lam0n0 () Int)
(declare-fun lam0n1 () Int)
(declare-fun lam0n2 () Int)
(declare-fun lam0n3 () Int)
(declare-fun Nl2CT1 () Int)
(declare-fun Nl2CT2 () Int)
(declare-fun lam1n0 () Int)
(declare-fun lam1n1 () Int)
(declare-fun lam1n2 () Int)
(declare-fun lam1n3 () Int)
(declare-fun lam2n0 () Int)
(declare-fun lam2n1 () Int)
(declare-fun lam2n2 () Int)
(declare-fun lam2n3 () Int)
(declare-fun lam3n0 () Int)
(declare-fun lam3n1 () Int)
(declare-fun lam3n2 () Int)
(declare-fun lam3n3 () Int)
(declare-fun lam3n4 () Int)
(declare-fun lam4n0 () Int)
(declare-fun lam4n1 () Int)
(declare-fun lam4n2 () Int)
(declare-fun lam4n3 () Int)
(declare-fun lam4n4 () Int)
(declare-fun main_x () Int)
(declare-fun main_y () Int)
(declare-fun main_z () Int)
(declare-fun undef4 () Int)
(declare-fun undef5 () Int)
(declare-fun undef6 () Int)
(declare-fun lam8n0 () Int)
(declare-fun lam8n1 () Int)
(declare-fun lam8n2 () Int)
(declare-fun lam8n3 () Int)
(declare-fun lam8n4 () Int)
(declare-fun lam8n5 () Int)
(declare-fun lam8n6 () Int)
(declare-fun lam8n7 () Int)
(declare-fun RFN1_CT () Int)
(declare-fun RFN1_main_x () Int)
(declare-fun RFN1_main_y () Int)
(declare-fun RFN1_main_z () Int)
(declare-fun lam6n0 () Int)
(declare-fun lam6n1 () Int)
(declare-fun lam6n2 () Int)
(declare-fun lam6n3 () Int)
(declare-fun lam6n4 () Int)
(declare-fun lam6n5 () Int)
(declare-fun lam6n6 () Int)
(declare-fun lam6n7 () Int)
(declare-fun lam7n0 () Int)
(declare-fun lam7n1 () Int)
(declare-fun lam7n2 () Int)
(declare-fun lam7n3 () Int)
(declare-fun lam7n4 () Int)
(declare-fun lam7n5 () Int)
(declare-fun lam7n6 () Int)
(declare-fun lam7n7 () Int)
(declare-fun lam5n0 () Int)
(declare-fun lam5n1 () Int)
(declare-fun lam5n2 () Int)
(declare-fun lam5n3 () Int)
(declare-fun lam5n4 () Int)
(declare-fun lam5n5 () Int)
(declare-fun lam5n6 () Int)
(declare-fun lam5n7 () Int)
(assert ( and ( <= ( - 1 ) Nl2main_x1 ) ( <= Nl2main_x1 1 ) ( <= ( - 1 ) Nl2main_y1 ) ( <= Nl2main_y1 1 ) ( <= ( - 1 ) Nl2main_z1 ) ( <= Nl2main_z1 1 ) ( <= ( - 1 ) Nl2main_x2 ) ( <= Nl2main_x2 1 ) ( <= ( - 1 ) Nl2main_y2 ) ( <= Nl2main_y2 1 ) ( <= ( - 1 ) Nl2main_z2 ) ( <= Nl2main_z2 1 ) ))
(assert ( or ( and ( >= lam0n0 0 ) ( >= lam0n1 0 ) ( >= lam0n2 0 ) ( >= lam0n3 0 ) ( > ( + ( * 1 lam0n0 ) ( * 1 lam0n1 ) ( * Nl2CT1 lam0n2 ) ( * Nl2CT2 lam0n3 ) ( - 1 ) ) 0 ) ( = ( + ( * ( - 1 ) lam0n0 ) ( * ( - 1 ) lam0n1 ) ( * Nl2main_x1 lam0n2 ) ( * Nl2main_x2 lam0n3 ) ) 0 ) ( = ( + ( * ( - 1 ) lam0n1 ) ( * Nl2main_y1 lam0n2 ) ( * Nl2main_y2 lam0n3 ) ) 0 ) ( = ( + ( * ( - 1 ) lam0n1 ) ( * Nl2main_z1 lam0n2 ) ( * Nl2main_z2 lam0n3 ) ) 0 ) ) ( and ( >= lam1n0 0 ) ( >= lam1n1 0 ) ( >= lam1n2 0 ) ( >= lam1n3 0 ) ( > ( + ( * 1 lam1n0 ) ( * 1 lam1n1 ) ( * Nl2CT1 lam1n2 ) ( * Nl2CT2 lam1n3 ) ( - 1 ( + ( + ( + Nl2CT1 ( * Nl2main_x1 0 ) ) ( * Nl2main_y1 0 ) ) ( * Nl2main_z1 1 ) ) ) ) 0 ) ( = ( + ( * ( - 1 ) lam1n0 ) ( * ( - 1 ) lam1n1 ) ( * Nl2main_x1 lam1n2 ) ( * Nl2main_x2 lam1n3 ) ( - ( + 0 ( * Nl2main_x1 1 ) ) ) ) 0 ) ( = ( + ( * ( - 1 ) lam1n1 ) ( * Nl2main_y1 lam1n2 ) ( * Nl2main_y2 lam1n3 ) ( - ( + ( + 0 ( * Nl2main_x1 1 ) ) ( * Nl2main_y1 1 ) ) ) ) 0 ) ( = ( + ( * ( - 1 ) lam1n1 ) ( * Nl2main_z1 lam1n2 ) ( * Nl2main_z2 lam1n3 ) ( - ( + ( + 0 ( * Nl2main_y1 1 ) ) ( * Nl2main_z1 1 ) ) ) ) 0 ) ) ))
(assert ( or ( and ( >= lam0n0 0 ) ( >= lam0n1 0 ) ( >= lam0n2 0 ) ( >= lam0n3 0 ) ( > ( + ( * 1 lam0n0 ) ( * 1 lam0n1 ) ( * Nl2CT1 lam0n2 ) ( * Nl2CT2 lam0n3 ) ( - 1 ) ) 0 ) ( = ( + ( * ( - 1 ) lam0n0 ) ( * ( - 1 ) lam0n1 ) ( * Nl2main_x1 lam0n2 ) ( * Nl2main_x2 lam0n3 ) ) 0 ) ( = ( + ( * ( - 1 ) lam0n1 ) ( * Nl2main_y1 lam0n2 ) ( * Nl2main_y2 lam0n3 ) ) 0 ) ( = ( + ( * ( - 1 ) lam0n1 ) ( * Nl2main_z1 lam0n2 ) ( * Nl2main_z2 lam0n3 ) ) 0 ) ) ( and ( >= lam2n0 0 ) ( >= lam2n1 0 ) ( >= lam2n2 0 ) ( >= lam2n3 0 ) ( > ( + ( * 1 lam2n0 ) ( * 1 lam2n1 ) ( * Nl2CT1 lam2n2 ) ( * Nl2CT2 lam2n3 ) ( - 1 ( + ( + ( + Nl2CT2 ( * Nl2main_x2 0 ) ) ( * Nl2main_y2 0 ) ) ( * Nl2main_z2 1 ) ) ) ) 0 ) ( = ( + ( * ( - 1 ) lam2n0 ) ( * ( - 1 ) lam2n1 ) ( * Nl2main_x1 lam2n2 ) ( * Nl2main_x2 lam2n3 ) ( - ( + 0 ( * Nl2main_x2 1 ) ) ) ) 0 ) ( = ( + ( * ( - 1 ) lam2n1 ) ( * Nl2main_y1 lam2n2 ) ( * Nl2main_y2 lam2n3 ) ( - ( + ( + 0 ( * Nl2main_x2 1 ) ) ( * Nl2main_y2 1 ) ) ) ) 0 ) ( = ( + ( * ( - 1 ) lam2n1 ) ( * Nl2main_z1 lam2n2 ) ( * Nl2main_z2 lam2n3 ) ( - ( + ( + 0 ( * Nl2main_y2 1 ) ) ( * Nl2main_z2 1 ) ) ) ) 0 ) ) ))
(assert ( and ( not ( <= main_x 0 ) ) ( not ( <= ( + main_x main_y main_z ) 0 ) ) ( = ( + main_x ( * ( - 1 ) undef4 ) ) 0 ) ( = ( + main_y ( * ( - 1 ) undef5 ) ) 0 ) ( = ( + main_z ( * ( - 1 ) undef6 ) ) 0 ) ( <= ( + Nl2CT1 ( * ( + 0 Nl2main_x1 ) main_x ) ( * ( + 0 Nl2main_y1 ) main_y ) ( * ( + 0 Nl2main_z1 ) main_z ) ) 0 ) ( <= ( + Nl2CT2 ( * ( + 0 Nl2main_x2 ) main_x ) ( * ( + 0 Nl2main_y2 ) main_y ) ( * ( + 0 Nl2main_z2 ) main_z ) ) 0 ) ))
(assert ( or ( and ( and ( >= lam8n0 0 ) ( >= lam8n1 0 ) ( >= lam8n2 0 ) ( >= lam8n3 0 ) ( >= lam8n4 0 ) ( >= lam8n5 0 ) ( >= lam8n6 0 ) ( >= lam8n7 0 ) ( > ( + ( * 50001 lam8n0 ) ( * 50001 lam8n1 ) ( * 50001 lam8n2 ) ( * 50001 lam8n3 ) ( * 50001 lam8n4 ) ( * 50001 lam8n5 ) ( * Nl2CT1 lam8n6 ) ( * Nl2CT2 lam8n7 ) ( - 1 ( - ( + ( + ( + RFN1_CT ( * RFN1_main_x 0 ) ) ( * RFN1_main_y 0 ) ) ( * RFN1_main_z 1 ) ) RFN1_CT ) ) ) 0 ) ( = ( + ( * ( - 1 ) lam8n0 ) ( * ( - 1 ) lam8n1 ) ( * ( - 1 ) lam8n2 ) ( * ( - 1 ) lam8n3 ) ( * Nl2main_x1 lam8n6 ) ( * Nl2main_x2 lam8n7 ) ( - ( - ( + 0 ( * RFN1_main_x 1 ) ) RFN1_main_x ) ) ) 0 ) ( = ( + ( * ( - 1 ) lam8n1 ) ( * ( - 1 ) lam8n2 ) ( * ( - 1 ) lam8n4 ) ( * Nl2main_y1 lam8n6 ) ( * Nl2main_y2 lam8n7 ) ( - ( - ( + ( + 0 ( * RFN1_main_x 1 ) ) ( * RFN1_main_y 1 ) ) RFN1_main_y ) ) ) 0 ) ( = ( + ( * 1 lam8n1 ) ( * ( - 1 ) lam8n2 ) ( * 1 lam8n3 ) ( * ( - 1 ) lam8n5 ) ( * Nl2main_z1 lam8n6 ) ( * Nl2main_z2 lam8n7 ) ( - ( - ( + ( + 0 ( * RFN1_main_y 1 ) ) ( * RFN1_main_z 1 ) ) RFN1_main_z ) ) ) 0 ) ) ( and ( and ( >= lam6n0 0 ) ( >= lam6n1 0 ) ( >= lam6n2 0 ) ( >= lam6n3 0 ) ( >= lam6n4 0 ) ( >= lam6n5 0 ) ( >= lam6n6 0 ) ( >= lam6n7 0 ) ( > ( + ( * 50001 lam6n0 ) ( * 50001 lam6n1 ) ( * 50001 lam6n2 ) ( * 50001 lam6n3 ) ( * 50001 lam6n4 ) ( * 50001 lam6n5 ) ( * Nl2CT1 lam6n6 ) ( * Nl2CT2 lam6n7 ) ( - 1 ( - RFN1_CT ) ) ) 0 ) ( = ( + ( * ( - 1 ) lam6n0 ) ( * ( - 1 ) lam6n1 ) ( * ( - 1 ) lam6n2 ) ( * ( - 1 ) lam6n3 ) ( * Nl2main_x1 lam6n6 ) ( * Nl2main_x2 lam6n7 ) ( - ( - RFN1_main_x ) ) ) 0 ) ( = ( + ( * ( - 1 ) lam6n1 ) ( * ( - 1 ) lam6n2 ) ( * ( - 1 ) lam6n4 ) ( * Nl2main_y1 lam6n6 ) ( * Nl2main_y2 lam6n7 ) ( - ( - RFN1_main_y ) ) ) 0 ) ( = ( + ( * 1 lam6n1 ) ( * ( - 1 ) lam6n2 ) ( * 1 lam6n3 ) ( * ( - 1 ) lam6n5 ) ( * Nl2main_z1 lam6n6 ) ( * Nl2main_z2 lam6n7 ) ( - ( - RFN1_main_z ) ) ) 0 ) ) ( and ( >= lam7n0 0 ) ( >= lam7n1 0 ) ( >= lam7n2 0 ) ( >= lam7n3 0 ) ( >= lam7n4 0 ) ( >= lam7n5 0 ) ( >= lam7n6 0 ) ( >= lam7n7 0 ) ( > ( + ( * 50001 lam7n0 ) ( * 50001 lam7n1 ) ( * 50001 lam7n2 ) ( * 50001 lam7n3 ) ( * 50001 lam7n4 ) ( * 50001 lam7n5 ) ( * Nl2CT1 lam7n6 ) ( * Nl2CT2 lam7n7 ) ( - 1 ( + ( - ( + ( + ( + RFN1_CT ( * RFN1_main_x 0 ) ) ( * RFN1_main_y 0 ) ) ( * RFN1_main_z 1 ) ) RFN1_CT ) 1 ) ) ) 0 ) ( = ( + ( * ( - 1 ) lam7n0 ) ( * ( - 1 ) lam7n1 ) ( * ( - 1 ) lam7n2 ) ( * ( - 1 ) lam7n3 ) ( * Nl2main_x1 lam7n6 ) ( * Nl2main_x2 lam7n7 ) ( - ( - ( + 0 ( * RFN1_main_x 1 ) ) RFN1_main_x ) ) ) 0 ) ( = ( + ( * ( - 1 ) lam7n1 ) ( * ( - 1 ) lam7n2 ) ( * ( - 1 ) lam7n4 ) ( * Nl2main_y1 lam7n6 ) ( * Nl2main_y2 lam7n7 ) ( - ( - ( + ( + 0 ( * RFN1_main_x 1 ) ) ( * RFN1_main_y 1 ) ) RFN1_main_y ) ) ) 0 ) ( = ( + ( * 1 lam7n1 ) ( * ( - 1 ) lam7n2 ) ( * 1 lam7n3 ) ( * ( - 1 ) lam7n5 ) ( * Nl2main_z1 lam7n6 ) ( * Nl2main_z2 lam7n7 ) ( - ( - ( + ( + 0 ( * RFN1_main_y 1 ) ) ( * RFN1_main_z 1 ) ) RFN1_main_z ) ) ) 0 ) ) ) ) ( and ( >= lam5n0 0 ) ( >= lam5n1 0 ) ( >= lam5n2 0 ) ( >= lam5n3 0 ) ( >= lam5n4 0 ) ( >= lam5n5 0 ) ( >= lam5n6 0 ) ( >= lam5n7 0 ) ( > ( + ( * 50001 lam5n0 ) ( * 50001 lam5n1 ) ( * 50001 lam5n2 ) ( * 50001 lam5n3 ) ( * 50001 lam5n4 ) ( * 50001 lam5n5 ) ( * Nl2CT1 lam5n6 ) ( * Nl2CT2 lam5n7 ) ( - 1 ) ) 0 ) ( = ( + ( * ( - 1 ) lam5n0 ) ( * ( - 1 ) lam5n1 ) ( * ( - 1 ) lam5n2 ) ( * ( - 1 ) lam5n3 ) ( * Nl2main_x1 lam5n6 ) ( * Nl2main_x2 lam5n7 ) ) 0 ) ( = ( + ( * ( - 1 ) lam5n1 ) ( * ( - 1 ) lam5n2 ) ( * ( - 1 ) lam5n4 ) ( * Nl2main_y1 lam5n6 ) ( * Nl2main_y2 lam5n7 ) ) 0 ) ( = ( + ( * 1 lam5n1 ) ( * ( - 1 ) lam5n2 ) ( * 1 lam5n3 ) ( * ( - 1 ) lam5n5 ) ( * Nl2main_z1 lam5n6 ) ( * Nl2main_z2 lam5n7 ) ) 0 ) ) ))
(check-sat)
(exit)
