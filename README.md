# Magnolia Image Tagging Module

Magnolia dedicated image tagging module which utilises Google-Image-Tagging 
capabilities but can be replaced with any other tool. On module startup it will query the 
'dam' workspace and find images which do not have tags already.

Afterwards It will communicate to provided image tagging library for the tags and store them
in the given node's 'imageTags' property.