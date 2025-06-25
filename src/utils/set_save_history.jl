#To set save history
function set_save_history!(hgf::HGF, save_history::Bool)
    hgf.save_history = save_history
end

function set_save_history!(agent::Agent{<:HGF}, save_history::Bool)
    agent.model_attributes.submodel.save_history = save_history
end
