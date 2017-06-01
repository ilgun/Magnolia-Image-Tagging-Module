/**
 * This file Copyright (c) 2017 Magnolia International
 * Ltd.  (http://www.magnolia-cms.com). All rights reserved.
 *
 *
 * This file is dual-licensed under both the Magnolia
 * Network Agreement and the GNU General Public License.
 * You may elect to use one or the other of these licenses.
 *
 * This file is distributed in the hope that it will be
 * useful, but AS-IS and WITHOUT ANY WARRANTY; without even the
 * implied warranty of MERCHANTABILITY or FITNESS FOR A
 * PARTICULAR PURPOSE, TITLE, or NONINFRINGEMENT.
 * Redistribution, except as permitted by whichever of the GPL
 * or MNA you select, is prohibited.
 *
 * 1. For the GPL license (GPL), you can redistribute and/or
 * modify this file under the terms of the GNU General
 * Public License, Version 3, as published by the Free Software
 * Foundation.  You should have received a copy of the GNU
 * General Public License, Version 3 along with this program;
 * if not, write to the Free Software Foundation, Inc., 51
 * Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 * 2. For the Magnolia Network Agreement (MNA), this file
 * and the accompanying materials are made available under the
 * terms of the MNA which accompanies this distribution, and
 * is available at http://www.magnolia-cms.com/mna.html
 *
 * Any modifications to this file must keep this entire header
 * intact.
 *
 */
package info.magnolia.image.tagging;


import info.magnolia.context.Context;
import info.magnolia.jcr.util.NodeUtil;
import info.magnolia.ml.GoogleImageTaggingService;
import info.magnolia.module.ModuleLifecycle;
import info.magnolia.module.ModuleLifecycleContext;

import java.util.Collection;
import java.util.List;

import javax.inject.Inject;
import javax.jcr.Node;
import javax.jcr.RepositoryException;
import javax.jcr.Session;

import org.apache.commons.lang3.StringUtils;
import org.apache.jackrabbit.commons.predicate.Predicate;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.collect.Lists;

/**
 * Magnolia module which queries {@link #DAM_WORKSPACE dam workspace} and tags images if they are not tagged already.
 */
public class ImageTaggingModule implements ModuleLifecycle {

    private static final Logger log = LoggerFactory.getLogger(ImageTaggingModule.class);
    private static final String IMAGE_TAGS_PROPERTY = "imageTags";
    private static final String DAM_WORKSPACE = "dam";

    private final Context context;
    private final GoogleImageTaggingService imageTaggingService;

    @Inject
    public ImageTaggingModule(Context context) {
        this.context = context;
        this.imageTaggingService = new GoogleImageTaggingService();
    }

    @Override
    public void start(ModuleLifecycleContext moduleLifecycleContext) {
        try {
            //TODO: Might want to use some threading here
            Session damSession = context.getJCRSession(DAM_WORKSPACE);
            List<Node> allImagesWithoutPresentTag = Lists.newArrayList(NodeUtil.collectAllChildren(damSession.getRootNode(),
                    notTaggedImagesPredicate()));

            // TODO: it's a bit pricey so limit it to 5 :)
            for (int i = 0; i < 5; i++) {
                Node imageNode = allImagesWithoutPresentTag.get(i);
                Collection<String> tags = imageTaggingService.processImage(imageNode.getNode("jcr:content").getProperty("jcr:data").getBinary().getStream());
                imageNode.setProperty(IMAGE_TAGS_PROPERTY, StringUtils.join(tags));
                imageNode.getSession().save();
            }
        } catch (RepositoryException e) {
            log.error("An error occurred while tagging images", e);
        }
    }

    /**
     * A {@link Predicate} for finding not tagged images.
     */
    private static Predicate notTaggedImagesPredicate() {
        return object -> {
            Node node = (Node) object;
            try {
                return !node.hasProperty(IMAGE_TAGS_PROPERTY) && node.getPrimaryNodeType().getName().equals("mgnl:asset");
            } catch (RepositoryException e) {
                e.printStackTrace();
            }
            return false;
        };
    }

    @Override
    public void stop(ModuleLifecycleContext moduleLifecycleContext) {

    }
}