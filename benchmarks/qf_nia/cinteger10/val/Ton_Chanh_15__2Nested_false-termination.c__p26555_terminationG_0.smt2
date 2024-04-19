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
(set-info :status unsat)
(declare-fun Nl2main_x1 () Int)
(declare-fun Nl2main_y1 () Int)
(declare-fun lam0n0 () Int)
(declare-fun lam0n1 () Int)
(declare-fun lam0n2 () Int)
(declare-fun Nl2CT1 () Int)
(declare-fun lam1n0 () Int)
(declare-fun lam1n1 () Int)
(declare-fun lam1n2 () Int)
(declare-fun lam2n0 () Int)
(declare-fun lam2n1 () Int)
(declare-fun lam2n2 () Int)
(declare-fun lam2n3 () Int)
(declare-fun main_x () Int)
(declare-fun main_y () Int)
(declare-fun undef3 () Int)
(declare-fun undef4 () Int)
(declare-fun lam6n0 () Int)
(declare-fun lam6n1 () Int)
(declare-fun lam6n2 () Int)
(declare-fun lam6n3 () Int)
(declare-fun lam6n4 () Int)
(declare-fun RFN1_CT () Int)
(declare-fun RFN1_main_x () Int)
(declare-fun RFN1_main_y () Int)
(declare-fun lam4n0 () Int)
(declare-fun lam4n1 () Int)
(declare-fun lam4n2 () Int)
(declare-fun lam4n3 () Int)
(declare-fun lam4n4 () Int)
(declare-fun lam5n0 () Int)
(declare-fun lam5n1 () Int)
(declare-fun lam5n2 () Int)
(declare-fun lam5n3 () Int)
(declare-fun lam5n4 () Int)
(declare-fun lam3n0 () Int)
(declare-fun lam3n1 () Int)
(declare-fun lam3n2 () Int)
(declare-fun lam3n3 () Int)
(declare-fun lam3n4 () Int)
(assert ( and ( <= ( - 1 ) Nl2main_x1 ) ( <= Nl2main_x1 1 ) ( <= ( - 1 ) Nl2main_y1 ) ( <= Nl2main_y1 1 ) ))
(assert ( or ( and ( >= lam0n0 0 ) ( >= lam0n1 0 ) ( >= lam0n2 0 ) ( > ( + ( * Nl2CT1 lam0n2 ) ( - 1 ) ) 0 ) ( = ( + ( * ( - 1 ) lam0n0 ) ( * ( - 1 ) lam0n1 ) ( * Nl2main_x1 lam0n2 ) ) 0 ) ( = ( + ( * ( - 1 ) lam0n1 ) ( * Nl2main_y1 lam0n2 ) ) 0 ) ) ( and ( >= lam1n0 0 ) ( >= lam1n1 0 ) ( >= lam1n2 0 ) ( > ( + ( * Nl2CT1 lam1n2 ) ( - 1 ( + ( + Nl2CT1 ( * Nl2main_x1 0 ) ) ( * Nl2main_y1 1 ) ) ) ) 0 ) ( = ( + ( * ( - 1 ) lam1n0 ) ( * ( - 1 ) lam1n1 ) ( * Nl2main_x1 lam1n2 ) ( - ( + 0 ( * Nl2main_x1 1 ) ) ) ) 0 ) ( = ( + ( * ( - 1 ) lam1n1 ) ( * Nl2main_y1 lam1n2 ) ( - ( + ( + 0 ( * Nl2main_x1 1 ) ) ( * Nl2main_y1 1 ) ) ) ) 0 ) ) ))
(assert ( and ( not ( <= ( + 1 main_x ) 0 ) ) ( not ( <= ( + 1 main_x main_y ) 0 ) ) ( = ( + main_x ( * ( - 1 ) undef3 ) ) 0 ) ( = ( + main_y ( * ( - 1 ) undef4 ) ) 0 ) ( <= ( + Nl2CT1 ( * ( + 0 Nl2main_x1 ) main_x ) ( * ( + 0 Nl2main_y1 ) main_y ) ) 0 ) ( <= ( * ( - 1 ) main_x ) 0 ) ( <= ( * ( - 1 ) main_x ) 0 ) ( <= ( * ( - 1 ) main_x ) 0 ) ))
(assert ( or ( and ( and ( >= lam6n0 0 ) ( >= lam6n1 0 ) ( >= lam6n2 0 ) ( >= lam6n3 0 ) ( >= lam6n4 0 ) ( > ( + ( * 50001 lam6n0 ) ( * 50001 lam6n1 ) ( * 50001 lam6n2 ) ( * 50001 lam6n3 ) ( * Nl2CT1 lam6n4 ) ( - 1 ( - ( + ( + RFN1_CT ( * RFN1_main_x 0 ) ) ( * RFN1_main_y 1 ) ) RFN1_CT ) ) ) 0 ) ( = ( + ( * ( - 1 ) lam6n0 ) ( * ( - 1 ) lam6n1 ) ( * ( - 1 ) lam6n2 ) ( * Nl2main_x1 lam6n4 ) ( - ( - ( + 0 ( * RFN1_main_x 1 ) ) RFN1_main_x ) ) ) 0 ) ( = ( + ( * 1 lam6n1 ) ( * ( - 1 ) lam6n2 ) ( * ( - 1 ) lam6n3 ) ( * Nl2main_y1 lam6n4 ) ( - ( - ( + ( + 0 ( * RFN1_main_x 1 ) ) ( * RFN1_main_y 1 ) ) RFN1_main_y ) ) ) 0 ) ) ( and ( and ( >= lam4n0 0 ) ( >= lam4n1 0 ) ( >= lam4n2 0 ) ( >= lam4n3 0 ) ( >= lam4n4 0 ) ( > ( + ( * 50001 lam4n0 ) ( * 50001 lam4n1 ) ( * 50001 lam4n2 ) ( * 50001 lam4n3 ) ( * Nl2CT1 lam4n4 ) ( - 1 ( - RFN1_CT ) ) ) 0 ) ( = ( + ( * ( - 1 ) lam4n0 ) ( * ( - 1 ) lam4n1 ) ( * ( - 1 ) lam4n2 ) ( * Nl2main_x1 lam4n4 ) ( - ( - RFN1_main_x ) ) ) 0 ) ( = ( + ( * 1 lam4n1 ) ( * ( - 1 ) lam4n2 ) ( * ( - 1 ) lam4n3 ) ( * Nl2main_y1 lam4n4 ) ( - ( - RFN1_main_y ) ) ) 0 ) ) ( and ( >= lam5n0 0 ) ( >= lam5n1 0 ) ( >= lam5n2 0 ) ( >= lam5n3 0 ) ( >= lam5n4 0 ) ( > ( + ( * 50001 lam5n0 ) ( * 50001 lam5n1 ) ( * 50001 lam5n2 ) ( * 50001 lam5n3 ) ( * Nl2CT1 lam5n4 ) ( - 1 ( + ( - ( + ( + RFN1_CT ( * RFN1_main_x 0 ) ) ( * RFN1_main_y 1 ) ) RFN1_CT ) 1 ) ) ) 0 ) ( = ( + ( * ( - 1 ) lam5n0 ) ( * ( - 1 ) lam5n1 ) ( * ( - 1 ) lam5n2 ) ( * Nl2main_x1 lam5n4 ) ( - ( - ( + 0 ( * RFN1_main_x 1 ) ) RFN1_main_x ) ) ) 0 ) ( = ( + ( * 1 lam5n1 ) ( * ( - 1 ) lam5n2 ) ( * ( - 1 ) lam5n3 ) ( * Nl2main_y1 lam5n4 ) ( - ( - ( + ( + 0 ( * RFN1_main_x 1 ) ) ( * RFN1_main_y 1 ) ) RFN1_main_y ) ) ) 0 ) ) ) ) ( and ( >= lam3n0 0 ) ( >= lam3n1 0 ) ( >= lam3n2 0 ) ( >= lam3n3 0 ) ( >= lam3n4 0 ) ( > ( + ( * 50001 lam3n0 ) ( * 50001 lam3n1 ) ( * 50001 lam3n2 ) ( * 50001 lam3n3 ) ( * Nl2CT1 lam3n4 ) ( - 1 ) ) 0 ) ( = ( + ( * ( - 1 ) lam3n0 ) ( * ( - 1 ) lam3n1 ) ( * ( - 1 ) lam3n2 ) ( * Nl2main_x1 lam3n4 ) ) 0 ) ( = ( + ( * 1 lam3n1 ) ( * ( - 1 ) lam3n2 ) ( * ( - 1 ) lam3n3 ) ( * Nl2main_y1 lam3n4 ) ) 0 ) ) ))
(check-sat)
(exit)
