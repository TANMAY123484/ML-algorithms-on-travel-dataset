rm(list = ls())

### Set directory 
##setwd("Proivde your data file location")

### Load Apollo library
library(apollo)

### Initialise code
apollo_initialise()

### Set core controls
apollo_control = list(
  modelName  ="MNL",
  modelDescr ="MNL model Estimation",
  indivID    ="ID"
)

# ################################################################# #
#### LOAD DATA AND APPLY ANY TRANSFORMATIONS                     ####
# ################################################################# #

database = read.csv("new_choice_dataset1.csv",header=TRUE)


# ################################################################# #
#### DEFINE MODEL PARAMETERS                                     ####
# ################################################################# #

### Vector of parameters, including any that are kept fixed in estimation
apollo_beta=c(asc_metro               = 0,
              asc_bus               = 0,
              asc_SR              = 0,
              asc_Auto              = 0,
              asc_bike              = 0,
              asc_car             = 0,
              asc_cycle             = 0,
              asc_walk              = 0,
              b_TT                = 0,
              b_TC=0,
              
              
              
              b_metro=0,
              b_SR=0,
              
              b_bike=0,
              b_car=0,
              b_cycle=0,
              b_walk=0,
              b_1=0,
              
              b_3=0,
              
              b_6=0,
              bv_1=0,
              
              bv_4=0,
              bv_5=0,
              
              b_inc1=0,
              b_inc2=0,
              b_inc3=0,
              b_inc4=0,
              b_inc5=0,
              b_inc6=0,
              b_inc7=0,
              b_inc8=0,
              
              b_p1=0,
              b_p2=0,
              
              b_p7=0,
              
              b_e1=0,
              b_e2=0,
              
              b_e4=0,
              b_e5=0,
              
              b_e7=0
              
              
)

### Vector with names (in quotes) of parameters to be kept fixed at their starting value in apollo_beta, use apollo_beta_fixed = c() if none
apollo_fixed = c("asc_metro","b_metro","b_1","b_inc1","bv_1","b_p1",
                 "b_e1")

# ################################################################# #
#### GROUP AND VALIDATE INPUTS                                   ####
# ################################################################# #

apollo_inputs = apollo_validateInputs()

# ################################################################# #
#### DEFINE MODEL AND LIKELIHOOD FUNCTION                        ####
# ################################################################# #
V = list()
apollo_probabilities=function(apollo_beta, apollo_inputs, functionality="estimate"){
  
  ### Attach inputs and detach after function exit
  apollo_attach(apollo_beta, apollo_inputs)
  on.exit(apollo_detach(apollo_beta, apollo_inputs))
  
  ### Create list of probabilities P
  P = list()
  
  ### List of utilities: these must use the same names as in mnl_settings, order is irrelevant
  
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
  
  ### Define settings for MNL model component
  mnl_settings = list(
    alternatives  = c(metro=1, bus=2, SR=3, Auto=4, bike=5, car=6, cycle=7, walk=8), 
    avail         = list(metro=av_metro, bus=av_bus, SR=av_SR, Auto=av_Auto, bike=av_bike, 
                         car=av_car, cycle=av_cycle, walk=av_walk),
    choiceVar     = Mode,
    V             = V
  )
  
  ### Compute probabilities using MNL model
  P[['model']] = apollo_mnl(mnl_settings, functionality)
  
  ### Take product across observation for same individual
  #P = apollo_panelProd(P, apollo_inputs, functionality)
  
  ### Prepare and return outputs of function
  P = apollo_prepareProb(P, apollo_inputs, functionality)
  return(P)
}

# ################################################################# #
#### MODEL ESTIMATION                                            ####
# ################################################################# #

model = apollo_estimate(apollo_beta, apollo_fixed, apollo_probabilities, apollo_inputs)

# ################################################################# #
#### MODEL OUTPUTS                                               ####
# ################################################################# #

# ----------------------------------------------------------------- #
#---- FORMATTED OUTPUT (TO SCREEN)                               ----
# ----------------------------------------------------------------- #

apollo_modelOutput(model)



predictions_base = apollo_prediction(model, 
                                     apollo_probabilities, 
                                     apollo_inputs,prediction_settings = list(),
                                     modelComponent = NA)
apollo_saveOutput(model)

apollo_inputs = apollo_validateInputs()
pred=apollo_prediction(model,apollo_probabilities,apollo_inputs)
#predictions_base=predictions_base[["at_estimates"]]
write.csv(pred,"pred_final.csv")

predictions_base = apollo_prediction(model, 
                                     apollo_probabilities, 
                                     apollo_inputs,prediction_settings = list(),
                                     modelComponent = NA)


database$tc_metro = database$tc_bus
database$tc_bike=1.2*database$tc_bike
database$tc_car=1.2*database$tc_car


database$av_metro=database$av_des
database$ED1=database$ED2
database$PD1=database$PD2
apollo_inputs   = apollo_validateInputs()
predictions_new = apollo_prediction(model, 
                                    apollo_probabilities, 
                                    apollo_inputs,prediction_settings = list(),
                                    modelComponent = NA)



### Compare predictions
change=(predictions_new-predictions_base)/predictions_base
### Not interested in chosen alternative now, 
### so drop last column
change=change[,-ncol(change)]
### Summary of changes (possible presence of NAs due to
### unavailable alternatives)
summary(change)