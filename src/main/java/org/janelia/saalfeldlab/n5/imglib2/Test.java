package org.janelia.saalfeldlab.n5.imglib2;

import java.io.IOException;
import java.util.Arrays;

import org.janelia.saalfeldlab.n5.DatasetAttributes;
import org.janelia.saalfeldlab.n5.N5;
import org.janelia.saalfeldlab.n5.N5Reader;
import org.janelia.saalfeldlab.n5.N5Writer;

import bdv.util.BdvFunctions;
import bdv.util.BdvOptions;
import bdv.util.BdvStackSource;
import bdv.util.RandomAccessibleIntervalMipmapSource;
import bdv.util.volatiles.SharedQueue;
import bdv.util.volatiles.VolatileViews;
import bdv.viewer.DisplayMode;
import mpicbg.spim.data.sequence.FinalVoxelDimensions;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.RealRandomAccessible;
import net.imglib2.cache.img.DiskCachedCellImg;
import net.imglib2.cache.img.DiskCachedCellImgFactory;
import net.imglib2.cache.img.DiskCachedCellImgOptions;
import net.imglib2.img.cell.CellGrid;
import net.imglib2.interpolation.randomaccess.NLinearInterpolatorFactory;
import net.imglib2.realtransform.AffineTransform3D;
import net.imglib2.realtransform.RealViews;
import net.imglib2.type.numeric.integer.UnsignedByteType;
import net.imglib2.type.numeric.integer.UnsignedShortType;
import net.imglib2.type.volatiles.VolatileUnsignedByteType;
import net.imglib2.type.volatiles.VolatileUnsignedShortType;
import net.imglib2.view.RandomAccessibleOnRealRandomAccessible;
import net.imglib2.view.Views;

public class Test
{

	public static void main1( final String[] args ) throws IOException
	{
		// davi
//		final String n5path = "/data/hanslovskyp/n5-tests/davi-toy-set";
//		final String dataset = "excerpt";

		// tomoko
//		final String n5path = "/data/hanslovskyp/tomoko/n5-test";
//		final String dataset = "z=0,512-bzip2";

		// nrs
		final String n5path = "/nrs/saalfeld/hanslovskyp/CutOn4-15-2013_ImagedOn1-27-2014/aligned/substacks/1300-3449/4000x2500+5172+1416/n5";
		final String dataset = "gzip";

		final N5Writer n5 = N5.openFSWriter( n5path );
		final DatasetAttributes attr = n5.getDatasetAttributes( dataset );
		final long[] dim = attr.getDimensions();
		final int[] cellSize = attr.getBlockSize();

		final CellGrid grid = new CellGrid( dim, cellSize );

		System.out.println( attr.getNumDimensions() + " " + attr.getDataType() + " " + attr.getCompressionType() + " " + Arrays.toString( attr.getDimensions() ) + " " + Arrays.toString( attr.getBlockSize() ) );

		final int numProc = Runtime.getRuntime().availableProcessors();

		final SharedQueue queue = new SharedQueue( numProc - 1 );

		final N5CellLoader< UnsignedByteType > loader = new N5CellLoader<>( n5, dataset, cellSize );

		final DiskCachedCellImgOptions options = DiskCachedCellImgOptions
				.options()
				.cellDimensions( cellSize )
				.dirtyAccesses( false )
				.maxCacheSize( 100 );

		final DiskCachedCellImgFactory< UnsignedByteType > factory = new DiskCachedCellImgFactory<>( options );

		final DiskCachedCellImg< UnsignedByteType, ? > img = factory.create( dim, new UnsignedByteType(), loader );

		final BdvStackSource< VolatileUnsignedByteType > bdv = BdvFunctions.show( VolatileViews.wrapAsVolatile( img, queue ), "volatile!" );
		bdv.getBdvHandle().getSetupAssignments().getMinMaxGroups().get( 0 ).setRange( 0, 255 );
		bdv.getBdvHandle().getViewerPanel().setDisplayMode( DisplayMode.SINGLE );

		final RealRandomAccessible< UnsignedByteType > transformSource = Views.interpolate( Views.extendBorder( img ), new NLinearInterpolatorFactory<>() );
		final AffineTransform3D transform = new AffineTransform3D();
		transform.setTranslation( 100, 200, 0 );
		final RandomAccessibleOnRealRandomAccessible< UnsignedByteType > transformed = Views.raster( RealViews.transformReal( transformSource, transform ) );
		final RandomAccessibleLoader< UnsignedByteType > transformLoader = new RandomAccessibleLoader<>( transformed );
		final DiskCachedCellImg< UnsignedByteType, ? > transformedCellImg = factory.create( dim, new UnsignedByteType(), transformLoader );
		BdvFunctions.show( VolatileViews.wrapAsVolatile( transformedCellImg, queue ), "transformed", BdvOptions.options().addTo( bdv ) );
		bdv.getBdvHandle().getSetupAssignments().getMinMaxGroups().get( 1 ).setRange( 0, 255 );
	}

	public static void main( final String[] args ) throws IOException
	{
		// nrs
		final String n5path = "/nrs/saalfeld/igor/161211_YFP2_comparison2/n5-export/c0";
		final N5Reader n5 = N5.openFSReader( n5path );

		@SuppressWarnings( "unchecked" )
		final RandomAccessibleInterval< UnsignedShortType >[] mipmapSources = new RandomAccessibleInterval[6];
		final int[][] mipmapScales = new int[ 6 ][ 3 ];

		final int[] downsampleFactors = new int[]{ 1, 1, 2, 4, 9, 17 };

		for ( int s = 0; s < 6; ++s )
		{
			mipmapSources[ s ] = N5Utils.openVolatile( n5, "s" + s );
			mipmapScales[ s ][ 0 ] = mipmapScales[ s ][ 1 ] = 1 << s;
			mipmapScales[ s ][ 2 ] = downsampleFactors[ s ];
		}

		final int numProc = Runtime.getRuntime().availableProcessors();
		final SharedQueue queue = new SharedQueue(Math.max(1, numProc / 2));

		final RandomAccessibleIntervalMipmapSource< UnsignedShortType > source =
				new RandomAccessibleIntervalMipmapSource< UnsignedShortType >(
						mipmapSources,
						new UnsignedShortType(),
						mipmapScales,
						new FinalVoxelDimensions( "um", 0.097, 0.097, 0.18 ),
						"buh" );

		BdvFunctions.show( source.asVolatile( new VolatileUnsignedShortType(), queue ), BdvOptions.options() );
	}

}
