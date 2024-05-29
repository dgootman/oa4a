.components.schemas |= with_entries(
    .value |= 
        if .properties.content.type == "string" then
            .properties.content.format = "password"
        elif .properties.content.oneOf then
            .properties.content.oneOf |= map(
                if .type == "string" then
                    .format = "password"
                else
                    .
                end
            )
        elif .properties.prompt.type == "string" then
            .properties.prompt.format = "password"
        elif .properties.b64_json.type == "string" then
            .properties.b64_json.format = "password"
        elif .properties.choices.items.properties.text then
            .properties.choices.items.properties.text.format = "password"
        elif .properties.prompt.oneOf then
            .properties.prompt.oneOf |= map(
                if .type == "string" then
                    .format = "password"
                elif .items.type == "string" then
                    .items.format = "password"
                else
                    .
                end
            )
        else
            .
        end
)