

# goals 

- we use fomc docs in order to do x -> makes people not listen
- we dont want to be doing series- creation + regression

- we can ellicit from an llm estimates of parameters whicjh go into a dsge / hank model...

- what we need to show is :
    - you can ellicit these in a sensible way 
    - how can you avoid 
    - before a big shock behaviour = regular + anticipatory behaviours - distinguish the two
    - also ensure no double counting 'say I said smth like there is a pandemic at the moment nut due to end soon' on top of a forward looking model' 
    - uk hank is a reasonable : forward looking over rational expectations, solving a forward looking thing with RE with stickyness
    - like to find out : roughly, std macro models are great at dealing with economic effects (eg how acro vars interact with each others, and how ppl react at the behavioural level) they do not deal with non economic effects very well ( eg regime shifts, deep beahiouvarl changes)
    - how do we do : step 1 : ABM of an LLM
    - DSGE model + LLM -> convert this into smth that goes inside a model. You cant just overwrite the core behavioural, instead modify the underlying beta / sigma etc iusing the llm
        - opt 1 : you plug in the llm-estimates ( as long as you are in the boundary, you are fine )
        - opt 2 : if shocks are anticipated / non-anticipated, solve the macro model with overlapping futures + LLMs
        - opt 3 : how much of that behaviour is driven by forward looking on the part of the LLM.
    -  How to measure ? -> 
        - can you prompt it differently ?
        - how mych of the thing it has broken down 
        - llm and household blocs have been trained to 'emulate a household' 
        - when X happens, is it a case we cen decompose the llms answer as A is macro B is just LLM. (ie B is behaviour above and beyond what is in the model ). 
    - Paper structure : 
        - **we can do macro + LLMs**
        - here is what we get if we assume the llm is doing it perfectly, here is what we assumeif it only gets the the behavioural correctly. 
        - not only one shock in first period, and see whether it settles, go through each period, how do we then plug it in the macro model. 
        - 
        - there are pbs : look I can get rid of the double counting
        - look I can get reasonable behaviour out of it. 
        - look I can get scarring from high inflation
        - putting in simulated news reports 
    - iterativeness ?
        - 




- ignoring the lookahead ?
# 