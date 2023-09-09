
V[['metro']]  = asc_metro  + b_TT*tt_metro+ b_TC*tc_metro+bv_1*(HH_Vehicles)+b_metro*(Gender)+b_1*(Age)+
  b_inc1*(HH_inc)+b_p1*(PD1)+b_e1*(ED1)
V[['bus']]  = asc_bus  + b_TT*tt_bus+ b_TC*tc_bus+b_inc2*(HH_inc)+b_p2*(PD1)+b_e2*(ED1)
V[['SR']] = asc_SR +b_TT*tt_SR+ b_TC*tc_SR+b_SR*(Gender)+b_3*(Age)+b_inc3*(HH_inc)
V[['Auto']] = asc_Auto + b_TT*tt_Auto+b_TC*tc_Auto+bv_4*(HH_Vehicles)+b_inc4*(HH_inc)+b_e4*(ED1)
V[['bike']] = asc_bike + b_TT*tt_bike+b_TC*tc_bike+bv_5*(HH_Vehicles)+b_bike*(Gender)+bv_5*(HH_Vehicles)+b_inc5*(HH_inc)+
  b_e5*(ED1)
V[['car']] = asc_car +b_TT*tt_car+ b_TC*tc_car+b_car*(Gender)+b_6*(Age)+b_inc6*(HH_inc)
V[['cycle']] = asc_cycle+b_TT*tt_cycle+b_cycle*(Gender)+b_inc7*(HH_inc)+b_p7*(PD1)+b_e7*(ED1)
V[['walk']] = asc_walk+b_TT*tt_walk+b_walk*(Gender)+b_inc8*(HH_inc)







d<-read.csv("new_estimate.csv")

V = matrix(data=NA, nrow=nrow(database), ncol=8)

for (i in 1:nrow(database)) {
  V[i,1] = d$asc_metro + d$b_TT*database$tt_metro[i] + d$b_TC*database$tc_metro[i] + d$bv_1*database$HH_Vehicles[i] + d$b_metro*database$Gender[i] + d$b_1*database$Age[i] + d$b_inc1*database$HH_inc[i] + d$b_p1*database$PD1[i] + d$b_e1*database$ED1[i]
  
  V[i,2] = d$asc_bus + d$b_TT*database$tt_bus[i] + d$b_TC*database$tc_bus[i] + d$b_inc2*database$HH_inc[i] + d$b_p2*database$PD1[i] + d$b_e2*database$ED1[i]
  
  V[i,3] = d$asc_SR + d$b_TT*database$tt_SR[i] + d$b_TC*database$tc_SR[i] + d$b_SR*database$Gender[i] + d$b_3*database$Age[i] + d$b_inc3*database$HH_inc[i]
  
  V[i,4] = d$asc_Auto + d$b_TT*database$tt_Auto[i] + d$b_TC*database$tc_Auto[i] + d$bv_4*database$HH_Vehicles[i] + d$b_inc4*database$HH_inc[i] + d$b_e4*database$ED1[i]
  
  V[i,5] = d$asc_bike + d$b_TT*database$tt_bike[i] + d$b_TC*database$tc_bike[i] + d$bv_5*database$HH_Vehicles[i] + d$b_bike*database$Gender[i] + d$bv_5*database$HH_Vehicles[i] + d$b_inc5*database$HH_inc[i] + d$b_e5*database$ED1[i]
  
  V[i,6] = d$asc_car + d$b_TT*database$tt_car[i] + d$b_TC*database$tc_car[i] + d$b_car*database$Gender[i] + d$b_6*database$Age[i] + d$b_inc6*database$HH_inc[i]
  
  V[i,7] = d$asc_cycle + d$b_TT*database$tt_cycle[i] + d$b_cycle*database$Gender[i] + d$b_inc7*database$HH_inc[i] + d$b_p7*database$PD1[i] + d$b_e7*database$ED1[i]
  
  V[i,8] = d$asc_walk + d$b_TT*database$tt_walk[i] + d$b_walk*database$Gender[i] + d$b_inc8*database$HH_inc[i]
}

print(V)
write.csv(V,"new_utility_values.csv")
