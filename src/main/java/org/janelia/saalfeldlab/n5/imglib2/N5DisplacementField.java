package org.janelia.saalfeldlab.n5.imglib2;

import java.io.IOException;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.stream.Collectors;
import java.util.stream.DoubleStream;
import java.util.stream.LongStream;

import org.janelia.saalfeldlab.n5.Compression;
import org.janelia.saalfeldlab.n5.DataType;
import org.janelia.saalfeldlab.n5.DatasetAttributes;
import org.janelia.saalfeldlab.n5.N5Reader;
import org.janelia.saalfeldlab.n5.N5Writer;

import net.imglib2.Cursor;
import net.imglib2.Interval;
import net.imglib2.IterableInterval;
import net.imglib2.RandomAccessible;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.RealPoint;
import net.imglib2.RealRandomAccessible;
import net.imglib2.converter.Converter;
import net.imglib2.converter.Converters;
import net.imglib2.img.array.ArrayImg;
import net.imglib2.img.array.ArrayImgFactory;
import net.imglib2.img.cell.CellGrid;
import net.imglib2.interpolation.InterpolatorFactory;
import net.imglib2.interpolation.randomaccess.NLinearInterpolatorFactory;
import net.imglib2.realtransform.AffineGet;
import net.imglib2.realtransform.AffineTransform;
import net.imglib2.realtransform.ExplicitInvertibleRealTransform;
import net.imglib2.realtransform.DeformationFieldTransform;
import net.imglib2.realtransform.InvertibleRealTransform;
import net.imglib2.realtransform.RealTransform;
import net.imglib2.realtransform.RealTransformSequence;
import net.imglib2.realtransform.RealViews;
import net.imglib2.realtransform.Scale;
import net.imglib2.realtransform.Scale2D;
import net.imglib2.realtransform.Scale3D;
import net.imglib2.realtransform.Translation;
import net.imglib2.transform.integer.MixedTransform;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.IntegerType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.integer.ByteType;
import net.imglib2.type.numeric.integer.IntType;
import net.imglib2.type.numeric.integer.LongType;
import net.imglib2.type.numeric.integer.ShortType;
import net.imglib2.type.numeric.integer.UnsignedByteType;
import net.imglib2.type.numeric.integer.UnsignedIntType;
import net.imglib2.type.numeric.integer.UnsignedLongType;
import net.imglib2.type.numeric.integer.UnsignedShortType;
import net.imglib2.type.numeric.real.DoubleType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.util.Intervals;
import net.imglib2.util.Util;
import net.imglib2.view.IntervalView;
import net.imglib2.view.MixedTransformView;
import net.imglib2.view.Views;
import net.imglib2.view.composite.CompositeIntervalView;
import net.imglib2.view.composite.GenericComposite;

/**
 * Class with helper methods for saving displacement field transformations as N5 datasets.
 * 
 * @author John Bogovic
 *
 */
public class N5DisplacementField
{
	public static final String MULTIPLIER_ATTR = "quantization_multiplier";
	public static final String AFFINE_ATTR = "affine";
	public static final String SPACING_ATTR = "spacing";
	public static final String FORWARD_ATTR = "dfield";
	public static final String INVERSE_ATTR = "invdfield";

	public static final String EXTEND_ZERO = "ext_zero";
	public static final String EXTEND_MIRROR = "ext_mirror";
	public static final String EXTEND_BORDER = "ext_border";

	public final static int[] PERMUTATION2D = new int[] { 2, 0, 1 };
	public final static int[] PERMUTATION3D = new int[] { 3, 0, 1, 2 };


    /**
     * Saves forward and inverse deformation fields into the default n5 datasets.
     *
     * @param n5Writer
     * @param affine
     * @param forwardDfield
     * @param fwdspacing the pixel spacing (resolution) of the forward deformation field
     * @param inverseDfield
     * @param invspacing the pixel spacing (resolution) of the inverse deformation field
     * @param blockSize 
     * @param compression
     */ 
	public static final <T extends NativeType<T> & RealType<T>> void save(
			final N5Writer n5Writer,
			final AffineGet affine,
			final RandomAccessibleInterval< T > forwardDfield,
			final double[] fwdspacing,
			final RandomAccessibleInterval< T > inverseDfield,
			final double[] invspacing,
			final int[] blockSize,
			final Compression compression ) throws IOException
	{
	    save( n5Writer, FORWARD_ATTR, affine, forwardDfield, fwdspacing, blockSize, compression );
	    save( n5Writer, INVERSE_ATTR, affine.inverse(), inverseDfield, invspacing, blockSize, compression );
	}

	public static < Q extends IntegerType< Q > > Converter< DoubleType, Q > quantizationConverter(
			final int nd, final double maxError, final Q type )
	{
		/* 
		 * To keep the max vector error below maxError, 
		 * the error per coordinate must be below m
		 */
		final double m = 2 * Math.sqrt( maxError * maxError / nd );
		return new Converter< DoubleType, Q >()
		{
			@Override
			public void convert( DoubleType input, Q output )
			{
				output.setInteger( Math.round( input.getRealDouble() / m ) );
			}
		};
	}

	/**
	 * Writes a {@link RealTransform} as a displacement field.
	 * 
	 * @param n5
	 * @param dataset 
	 * @param spatialBlockSize the block size
	 * @param compression 
	 * @param transform
	 * @param pixelToPhysical
	 * @param interval
	 * @param quantizationType
	 * @param maxError
	 * @param threadPool
	 * @throws IOException
	 */
	public static final <T extends NativeType<T> > void save(
			final N5Writer n5,
			final String dataset,
			final int[] spatialBlockSize,
			final Compression compression,
			final RealTransform transform,
			final AffineGet pixelToPhysical,
			final Interval interval,
			final T quantizationType,
			final double maxError,
			ExecutorService threadPool ) throws IOException
	{
		final int ndims = transform.numSourceDimensions();

		int[] permutation;
		if( ndims == 2 )
			permutation = PERMUTATION2D;
		else
			permutation = PERMUTATION3D;

		final long[] spatialDimensions = interval.dimensionsAsLongArray();
		final long[] outputDimensions = new long[ ndims + 1];
		final int[] blockSize = new int[ ndims + 1];
		outputDimensions[ 0 ] = ndims;
		blockSize[ 0 ] = ndims;
		for( int i = 0; i < ndims; i++ )
		{
			outputDimensions[ i + 1 ] = spatialDimensions[ i ];
			blockSize[ i + 1 ] = spatialBlockSize[ i ];
		}

		final CellGrid outputCellGrid = new CellGrid( outputDimensions, blockSize );
		final long numBlocks = Intervals.numElements( outputCellGrid.getGridDimensions() );
		final List< Long > blockIndexes = LongStream.range( 0, numBlocks ).boxed().collect( Collectors.toList() );
		ArrayImgFactory< T > blockFactory = new ArrayImgFactory<>( quantizationType );

		final DataType datatype;
		final boolean isQuantized;
		if( quantizationType instanceof ByteType )
		{
			datatype = DataType.INT8;
			isQuantized = true;
		}
		else if( quantizationType instanceof ShortType )
		{
			datatype = DataType.INT16;
			isQuantized = true;
		}
		else if( quantizationType instanceof IntType )
		{
			datatype = DataType.INT32;
			isQuantized = true;
		}
		else if( quantizationType instanceof FloatType )
		{
			datatype = DataType.FLOAT32;
			isQuantized = false;
		}
		else if( quantizationType instanceof DoubleType )
		{
			datatype = DataType.FLOAT64;
			isQuantized = false;
		}
		else
		{
			System.err.println( "Type not yet supported" );
			return;
		}

		try
		{
			n5.createDataset( dataset,
					new DatasetAttributes( outputDimensions, blockSize, datatype, compression ));
		}
		catch ( IOException e1 )
		{
			System.err.println( "Could not create dataset" );
			e1.printStackTrace();
			return;
		}

		for( final long i : blockIndexes )
		{
			threadPool.submit( new Runnable()
			{
				@SuppressWarnings( "unchecked" )
				@Override
				public void run()
				{
					final RealTransform transformCopy = transform.copy();

					final CellGrid cellGrid = new CellGrid( outputDimensions, blockSize );
					final long[] blockGridPosition = new long[ cellGrid.numDimensions() ];
					cellGrid.getCellGridPositionFlat( i, blockGridPosition );

					final long[] targetMin = new long[ ndims + 1 ];
					final int[] cellDimensions = new int[ ndims + 1 ];
					cellGrid.getCellDimensions( blockGridPosition, targetMin, cellDimensions );

					final double[] min = new double[ ndims ];
					for( int i = 0; i < ndims; i++ )
					{
						min[ i ] = targetMin[ i + 1 ];
					}
					final double[] minBlock = new double[ ndims ];
					pixelToPhysical.apply( min, minBlock );

					AffineTransform pixelToPhysicalBlock = new AffineTransform( ndims );
					pixelToPhysicalBlock.concatenate( pixelToPhysical );
					pixelToPhysicalBlock.preConcatenate( new Translation( minBlock ));

					ArrayImg< T, ? > blockImg = blockFactory.create( cellDimensions );
					IntervalView< T > blockPermuted = N5DisplacementField.permute( blockImg, permutation );

					if( isQuantized )
						transformToDeformationQuantized( 
								transformCopy, (RandomAccessibleInterval <? extends IntegerType<?>>) blockPermuted, pixelToPhysicalBlock, maxError );
					else 
						transformToDeformation( transformCopy, 
								(RandomAccessibleInterval <? extends RealType<?>>) blockPermuted, pixelToPhysicalBlock );

					T type = quantizationType.copy();
					// write the block
					try
					{
						N5Utils.saveNonEmptyBlock( blockImg, n5, dataset, blockGridPosition, type );
					}
					catch ( IOException e )
					{
						e.printStackTrace();
					}
				}
			});
		}
	}

	public static final <T extends RealType<T>> void transformToDeformation(
			final RealTransform transform,
			final RandomAccessibleInterval<T> deformationField )
	{
		double[] s = DoubleStream.iterate( 1, x -> x ).limit( transform.numSourceDimensions() ).toArray();
		transformToDeformation( transform, deformationField,
				new Scale( s ));
	}

	/**
	 * Stores the given {@link RealTransform} into the given deformation field.
	 * 
	 * @param transform the transformation
	 * @param deformationField the deformation field
	 * @param pixelToPhysical the transform from discrete coordinates to physical coordinates
	 */
	public static final <T extends RealType<T>> void transformToDeformation(
			final RealTransform transform,
			final RandomAccessibleInterval<T> deformationField,
			final AffineGet pixelToPhysical )
	{
		assert deformationField.numDimensions() == ( transform.numSourceDimensions() + 1 );
		assert deformationField.dimension( deformationField.numDimensions() - 1 ) >= transform.numSourceDimensions();

		final int N = transform.numSourceDimensions();
		final RealPoint pt = new RealPoint( transform.numTargetDimensions() );
		final RealPoint ptXfm = new RealPoint( transform.numTargetDimensions() );

		final CompositeIntervalView< T, ? extends GenericComposite< T > > col = Views.collapse( deformationField );
		final Cursor< ? extends GenericComposite< T > > c = Views.flatIterable( col ).cursor();
		while ( c.hasNext() )
		{
			final GenericComposite< T > displacementVector = c.next();

			// transform the location of the cursor
			// and store the displacement
			pixelToPhysical.apply( c, pt );
			transform.apply( pt, ptXfm );

			for ( int i = 0; i < N; i++ )
				displacementVector.get( i ).setReal( ptXfm.getDoublePosition( i ) - pt.getDoublePosition( i ) ); 
		}
	}

	/**
	 * Stores the given {@link RealTransform} into the given deformation field.
	 * 
	 * @param transform the transformation
	 * @param deformationField the deformation field
	 * @param pixelToPhysical the transform from discrete coordinates to physical coordinates
	 */
	public static final <Q extends IntegerType<Q>> void transformToDeformationQuantized(
			final RealTransform transform,
			final RandomAccessibleInterval<Q> deformationField,
			final AffineGet pixelToPhysical,
			final Converter< DoubleType, Q > quantizer )
	{
		assert deformationField.numDimensions() == ( transform.numSourceDimensions() + 1 );
		assert deformationField.dimension( deformationField.numDimensions() - 1 ) >= transform.numSourceDimensions();

		final int N = transform.numSourceDimensions();
		final RealPoint pt = new RealPoint( transform.numTargetDimensions() );
		final RealPoint ptXfm = new RealPoint( transform.numTargetDimensions() );

		final CompositeIntervalView< Q, ? extends GenericComposite< Q > > col = Views.collapse( deformationField );
		final Cursor< ? extends GenericComposite< Q > > c = Views.flatIterable( col ).cursor();

		DoubleType displacement = new DoubleType();
		Q value = Util.getTypeFromInterval( deformationField ).copy();
		while ( c.hasNext() )
		{
			final GenericComposite< Q > displacementVector = c.next();

			// transform the location of the cursor
			// and store the displacement
			pixelToPhysical.apply( c, pt );
			transform.apply( pt, ptXfm );

			for ( int i = 0; i < N; i++ )
			{
				displacement.setReal( ptXfm.getDoublePosition( i ) - pt.getDoublePosition( i ) );
				quantizer.convert( displacement, value );
				displacementVector.get( i ).set( value );
			}
		}
	}

	public static final <Q extends IntegerType<Q>> void transformToDeformationQuantized(
			final RealTransform transform,
			final RandomAccessibleInterval<Q> deformationField,
			final AffineGet pixelToPhysical,
			final double maxError )
	{
		transformToDeformationQuantized( transform, deformationField,
				pixelToPhysical,
				quantizationConverter( transform.numSourceDimensions(), maxError, 
						Util.getTypeFromInterval( deformationField )));
	}

	public static final <Q extends IntegerType<Q>> void trasformToDeformationQuantized(
			final RealTransform transform,
			final RandomAccessibleInterval<Q> deformationField,
			final double maxError )
	{
		double[] s = DoubleStream.iterate( 1, x -> x ).limit( transform.numSourceDimensions() ).toArray();
		transformToDeformationQuantized( transform, deformationField,
				new Scale( s ),
				quantizationConverter( transform.numSourceDimensions(), maxError, 
						Util.getTypeFromInterval( deformationField )));
	}

    /**
     * Saves an affine transform and deformation field into a specified n5 dataset.
     *
     *
     * @param n5Writer
     * @param dataset
     * @param affine
     * @param dfield
     * @param spacing the pixel spacing (resolution) of the deformation field
     * @param blockSize
     * @param compression
     */
	public static final <T extends NativeType<T> & RealType<T>> void save(
			final N5Writer n5Writer,
			final String dataset,
			final AffineGet affine,
			final RandomAccessibleInterval< T > dfield,
			final double[] spacing,
			final int[] blockSize,
			final Compression compression ) throws IOException
	{
		N5Utils.save( dfield, n5Writer, dataset, blockSize, compression );

        if( affine != null )
            saveAffine( affine, n5Writer, dataset );

		if( spacing != null )
			n5Writer.setAttribute( dataset, SPACING_ATTR, spacing );
	}

    /**
     * Saves an affine transform and quantized deformation field into a specified n5 dataset.
     *
     * The deformation field here is saved as an {@link IntegerType}
     * which could compress better in some cases.  The multiplier from
     * original values to compressed values is chosen as the smallest
     * value that keeps the error (L2) between quantized and original vectors. 
     *
     * @param n5Writer
     * @param dataset
     * @param affine
     * @param dfield
     * @param spacing
     * @param blockSize
     * @param compression
     * @param outputType
     * @param maxError
     */
	public static final <T extends NativeType<T> & RealType<T>, Q extends NativeType<Q> & IntegerType<Q>> void save(
			final N5Writer n5Writer,
			final String dataset,
			final AffineGet affine,
			final RandomAccessibleInterval< T > dfield,
			final double[] spacing,
			final int[] blockSize,
			final Compression compression, 
			final Q outputType, 
			final double maxError ) throws Exception
	{
		
		saveQuantized( n5Writer, dataset, dfield, blockSize, compression, outputType, maxError );
		saveAffine( affine, n5Writer, dataset );
		if( spacing != null )
			n5Writer.setAttribute( dataset, SPACING_ATTR, spacing );
	}

    /**
     * Saves an affine transform as an attribute associated with an n5
     * dataset.
     *
     * @param affine
     * @param n5Writer
     * @param dataset
     */
	public static final void saveAffine(
			final AffineGet affine,
			final N5Writer n5Writer,
			final String dataset ) throws IOException
	{
		if( affine != null )
			n5Writer.setAttribute( dataset, AFFINE_ATTR,  affine.getRowPackedCopy() );
	}

    /**
     * Saves an affine transform and quantized deformation field into a specified n5 dataset.
     *
     * The deformation field here is saved as an {@link IntegerType}
     * which could compress better in some cases.  The multiplier from
     * original values to compressed values is chosen as the smallest
     * value that keeps the error (L2) between quantized and original vectors. 
     *
     * @param n5Writer
     * @param dataset
     * @param source
     * @param blockSize
     * @param compression
     * @param outputType
     * @param maxError
     */
	public static final <T extends RealType<T>, Q extends NativeType<Q> & IntegerType<Q>> void saveQuantized(
			final N5Writer n5Writer,
			final String dataset,
			final RandomAccessibleInterval<T> source,
			final int[] blockSize,
			final Compression compression,
			final Q outputType,
            final double maxError ) throws Exception
	{
		/* 
		 * To keep the max vector error below maxError, 
		 * the error per coordinate must be below m
		 */
		int nd = ( source.numDimensions() - 1 ); // vector field source has num dims + 1
		double m = 2 * Math.sqrt( maxError * maxError / nd );

		RandomAccessibleInterval< T > source_permuted = vectorAxisFirst( source );
		RandomAccessibleInterval< Q > source_quant = Converters.convert(
				source_permuted, 
				new Converter<T, Q>()
				{
					@Override
					public void convert(T input, Q output)
					{
						output.setInteger( Math.round( input.getRealDouble() / m ));
					}
				}, 
				outputType.copy());

        N5Utils.save( source_quant, n5Writer, dataset, blockSize, compression);
		n5Writer.setAttribute( dataset, MULTIPLIER_ATTR, m );
	}

    /**
     * Opens an {@link InvertibleRealTransform} from an n5 object.  Uses the provided datasets as the forward and 
     * inverse transformations.
     *
     * @param n5
     * @param forwardDataset
     * @param inverseDataset
     * @param defaultType
     * @param interpolator
     * @return the invertible transformation
     */
	public static final <T extends RealType<T> & NativeType<T>> ExplicitInvertibleRealTransform openInvertible( 
			final N5Reader n5,
			final String forwardDataset,
			final String inverseDataset,
			final T defaultType,
			final InterpolatorFactory< T, RandomAccessible<T> > interpolator ) throws Exception
	{
		if( !n5.datasetExists( forwardDataset ))
		{
			System.err.println( "dataset : " + forwardDataset + " does not exist.");
			return null;
		}

		if( !n5.datasetExists( inverseDataset ))
		{
			System.err.println( "dataset : " + inverseDataset + " does not exist.");
			return null;
		}

		return new ExplicitInvertibleRealTransform(
                open( n5, forwardDataset, false, defaultType, interpolator),
                open( n5, inverseDataset, true, defaultType, interpolator));
	}

    /**
     * Opens an {@link InvertibleRealTransform} from an n5 object using default datasets and linear interpolation.
     *
     * @param n5
     * @return the invertible transformation
     */
	public static final <T extends RealType<T> & NativeType<T>> ExplicitInvertibleRealTransform openInvertible( 
			final N5Reader n5 ) throws Exception
	{
		return openInvertible( n5, FORWARD_ATTR, INVERSE_ATTR, new DoubleType(), new NLinearInterpolatorFactory<>()  );
	}

    /**
     * Opens an {@link InvertibleRealTransform} from an multi-scale n5 object, at the specified level,
     * using linear interpolation.
     *
     * @param n5
     * @param level 
     * @return the invertible transformation
     */
	public static final <T extends RealType<T> & NativeType<T>> ExplicitInvertibleRealTransform openInvertible( 
			final N5Reader n5,
			final int level ) throws Exception
	{
		return openInvertible( n5, "/"+level+"/"+FORWARD_ATTR, "/"+level+"/"+INVERSE_ATTR, new DoubleType(), new NLinearInterpolatorFactory<>()  );
	}

    /**
     * Opens an {@link InvertibleRealTransform} from an n5 object if
     * possible.
     *
     * @param n5
     * @param defaultType
     * @param interpolator
     * @return the invertible transformation
     */
	public static final <T extends RealType<T> & NativeType<T>> ExplicitInvertibleRealTransform openInvertible( 
			final N5Reader n5,
			final T defaultType,
			final InterpolatorFactory< T, RandomAccessible<T> > interpolator ) throws Exception
	{
		return openInvertible( n5, FORWARD_ATTR, INVERSE_ATTR, defaultType, interpolator );
	}

    /**
     * Opens a {@link RealTransform} from an n5 dataset as a
     * displacement field.  The resulting transform is the concatenation
     * of an affine transform and a {@link DeformationFieldTransform}.
     *
     * @param n5
     * @param dataset
     * @param inverse
     * @param defaultType
     * @param interpolator
     * @return the transformation
     */
	public static final <T extends RealType<T> & NativeType<T>> RealTransform open( 
			final N5Reader n5,
			final String dataset,
			final boolean inverse,
			final T defaultType,
			final InterpolatorFactory< T, RandomAccessible<T> > interpolator ) throws Exception
	{

		AffineGet affine = openAffine( n5, dataset );

		DeformationFieldTransform< T > dfield = new DeformationFieldTransform<>(
				openCalibratedField( n5, dataset, interpolator, defaultType ));

		if( affine != null )
		{
			RealTransformSequence xfmSeq = new RealTransformSequence();
			if( inverse )
			{
				xfmSeq.add( affine );
				xfmSeq.add( dfield );
			}
			else
			{
				xfmSeq.add( dfield );
				xfmSeq.add( affine );
			}
			return xfmSeq;
		}
		else
		{
			return dfield;
		}
	}

    /**
     * Returns an {@link AffineGet} transform from pixel space to
     * physical space, for the given n5 dataset, if present, null
     * otherwise.
     *
     * @param n5
     * @param dataset
     * @return the affine transform
     */
	public static final AffineGet openPixelToPhysical( final N5Reader n5, final String dataset ) throws Exception
	{
		double[] spacing = n5.getAttribute( dataset, SPACING_ATTR, double[].class );
		if ( spacing == null )
			return null;

		// have to bump the dimension up by one to apply it to the displacement field
		int N = spacing.length;
		final AffineGet affineMtx;
		if ( N == 1 )
			affineMtx = new Scale( spacing[ 0 ] );
		else if ( N == 2 )
			affineMtx = new Scale2D( spacing );
		else if ( N == 3 )
			affineMtx = new Scale3D( spacing );
		else
			return null;

		return affineMtx;
	}

    /**
     * Returns and affine transform stored as an attribute in an n5 dataset.
     *
     * @param n5
     * @param dataset
     * @return the affine
     */
	public static final AffineGet openAffine( final N5Reader n5, final String dataset ) throws Exception
	{
		double[] affineMtxRow = n5.getAttribute( dataset, AFFINE_ATTR, double[].class );
		if ( affineMtxRow == null )
			return null;

		int N = affineMtxRow.length;
		final AffineTransform affineMtx;
		if ( N == 2 )
			affineMtx = new AffineTransform( 1 );
		else if ( N == 6 )
			affineMtx = new AffineTransform( 2 );
		else if ( N == 12 )
			affineMtx = new AffineTransform( 3 );
		else
			return null;

		affineMtx.set( affineMtxRow );
		return affineMtx;
	}

    /**
     * Returns a deformation field from the given n5 dataset.
     *
     * If the data is an {@link IntegerType}, returns an un-quantized
     * view of the dataset, otherwise, returns the raw {@link
     * RandomAccessibleInterval}.
     *
     * @param n5
     * @param dataset
     * @param defaultType
     * @return the deformation field as a RandomAccessibleInterval
     */
	@SuppressWarnings( "unchecked" )
	public static final <T extends NativeType<T> & RealType<T>, Q extends NativeType<Q> & IntegerType<Q>> RandomAccessibleInterval< T > openField( 
			final N5Reader n5,
			final String dataset,
			final T defaultType ) throws Exception
	{
		final DatasetAttributes attributes = n5.getDatasetAttributes(dataset);
		switch (attributes.getDataType()) {
		case INT8:
			return openQuantized( n5, dataset, (Q)new ByteType(), defaultType );
		case UINT8:
			return openQuantized( n5, dataset, (Q)new UnsignedByteType(), defaultType );
		case INT16:
			return openQuantized( n5, dataset, (Q)new ShortType(), defaultType );
		case UINT16:
			return openQuantized( n5, dataset, (Q)new UnsignedShortType(), defaultType );
		case INT32:
			return openQuantized( n5, dataset, (Q)new IntType(), defaultType );
		case UINT32:
			return openQuantized( n5, dataset, (Q)new UnsignedIntType(), defaultType );
		case INT64:
			return openQuantized( n5, dataset, (Q)new LongType(), defaultType );
		case UINT64:
			return openQuantized( n5, dataset, (Q)new UnsignedLongType(), defaultType );
		default:
			return openRaw( n5, dataset, defaultType );
		}
	}

    /**
     * Returns a deformation field in physical coordinates as a {@link
     * RealRandomAccessible} from an n5 dataset.
     *
     * Internally, opens the given n5 dataset as a {@link
     * RandomAccessibleInterval}, un-quantizes if necessary, uses
     * the input {@link InterpolatorFactory} for interpolation, and
     * transforms to physical coordinates using the pixel spacing stored
     * in the "spacing" attribute, if present.
     *
     * @param n5
     * @param dataset
     * @param interpolator
     * @param defaultType
     * @return the deformation field as a RealRandomAccessible
     */
	public static < T extends NativeType< T > & RealType< T > >  RealRandomAccessible< T >[] openCalibratedField(
			final N5Reader n5, final String dataset,
			final InterpolatorFactory< T, RandomAccessible< T > > interpolator, 
			final T defaultType ) throws Exception
	{
		return openCalibratedField( n5, dataset, interpolator, EXTEND_BORDER, defaultType );
	}

    /**
     * Returns a deformation field in physical coordinates as a {@link
     * RealRandomAccessible} from an n5 dataset.
     *
     * Internally, opens the given n5 dataset as a {@link
     * RandomAccessibleInterval}, un-quantizes if necessary, uses
     * the input {@link InterpolatorFactory} for interpolation, and
     * transforms to physical coordinates using the pixel spacing stored
     * in the "spacing" attribute, if present.
     *
     * @param n5 the n5 reader
     * @param dataset the n5 dataset
     * @param interpolator the type of interpolation 
     * @param extensionType the type of out-of-bounds extension
     * @param defaultType the type
     * @return the deformation field as a RealRandomAccessible
     */
	public static < T extends NativeType< T > & RealType< T > > RealRandomAccessible< T >[] openCalibratedField(
			final N5Reader n5, final String dataset,
			final InterpolatorFactory< T, RandomAccessible< T > > interpolator, 
			final String extensionType,
			final T defaultType ) throws Exception
	{
		RandomAccessibleInterval< T > dfieldRai = openField( n5, dataset, defaultType );
		if ( dfieldRai == null )
		{
			return null;
		}

		RandomAccessibleInterval< T > dfieldRaiPerm = vectorAxisLast( dfieldRai );

		// num spatial dimensions
		final int nd = dfieldRaiPerm.numDimensions() - 1;
		@SuppressWarnings( "unchecked" )
		RealRandomAccessible<T>[] displacements = new RealRandomAccessible[ nd ];

		for( int i = 0; i < nd; i++ )
		{
			IntervalView< T > coordDist = Views.hyperSlice( dfieldRaiPerm, nd, i );
			RealRandomAccessible< T > dfieldReal = null;
			if( extensionType.equals( EXTEND_MIRROR )){
				dfieldReal = Views.interpolate( Views.extendMirrorDouble( coordDist ), interpolator );
			}
			else if( extensionType.equals( EXTEND_BORDER )){
				dfieldReal = Views.interpolate( Views.extendBorder( coordDist ), interpolator );
			}
			else {
				dfieldReal = Views.interpolate( Views.extendZero( coordDist ), interpolator );
			}

			final AffineGet pix2Phys = openPixelToPhysical( n5, dataset );
			if ( pix2Phys != null )
				displacements[ i ] = RealViews.affine( dfieldReal, pix2Phys );
			else
				displacements[ i ] = dfieldReal;
		}

		return displacements;
	}

    /**
     * Opens a transform from an n5 dataset using linear interpolation
     * for the deformation field.
     *
     * @param n5
     * @param dataset
     * @param inverse
     * @return the transform
     */
	public static < T extends NativeType< T > & RealType< T > > RealTransform open(
			final N5Reader n5, final String dataset, boolean inverse ) throws Exception
	{
		return open( n5, dataset, inverse, new FloatType(), new NLinearInterpolatorFactory<FloatType>());
	}

    /**
     * Returns a deformation field as a {@link RandomAccessibleInterval}, ensuring that
     * the vector is stored in the last dimension.
     *
     * @param n5
     * @param dataset
     * @param defaultType
     * @return the deformation field
     */
	public static final < T extends RealType<T> & NativeType<T> > RandomAccessibleInterval< T > openRaw(
			final N5Reader n5,
			final String dataset,
			final T defaultType ) throws Exception
	{
        RandomAccessibleInterval< T > src = N5Utils.open( n5, dataset, defaultType  );
        return vectorAxisLast( src );
	}

    /**
     * Open a quantized (integer) {@link RandomAccessibleInterval} from an n5
     * dataset.
     *
     * @param n5
     * @param dataset
     * @param defaultQuantizedType
     * @param defaultType
     * @return the un-quantized data
     */
	public static final <Q extends RealType<Q> & NativeType<Q>, T extends RealType<T>> RandomAccessibleInterval< T > openQuantized(
			final N5Reader n5,
			final String dataset,
			final Q defaultQuantizedType,
			final T defaultType ) throws Exception
	{
        RandomAccessibleInterval< Q > src = N5Utils.open( n5, dataset, defaultQuantizedType  );
        
        // get the factor going from quantized to original values
        Double mattr = n5.getAttribute( dataset, MULTIPLIER_ATTR, Double.TYPE );
        final double m;
        if( mattr != null )
        	m = mattr.doubleValue();
        else
        	m = 1.0;

        RandomAccessibleInterval< Q > src_perm = vectorAxisLast( src );
        RandomAccessibleInterval< T > src_converted = Converters.convert(
        	src_perm, 
			new Converter<Q, T>()
			{
				@Override
				public void convert(Q input, T output) {
					output.setReal( input.getRealDouble() * m );
				}
			}, 
			defaultType.copy());
 
        return src_converted;
	}

    /**
     * Returns a deformation field as a {@link RandomAccessibleInterval}
     * with the vector stored in the last dimension.
     *
     * @param source
     * @return the possibly permuted deformation field
     * @throws Exception
     */
	public static final < T extends RealType< T > > RandomAccessibleInterval< T > vectorAxisLast( RandomAccessibleInterval< T > source ) throws Exception
	{
		final int n = source.numDimensions();

		if ( source.dimension( n - 1 ) == (n - 1) )
			return source;
		else if ( source.dimension( 0 ) == (n - 1) )
		{
			final int[] component = new int[ n ];
			component[ 0 ] = n - 1;
			for ( int i = 1; i < n; ++i )
				component[ i ] = i - 1;

			return permute( source, component );
		}

		throw new Exception( 
				String.format( "Displacement fields must store vector components in the first or last dimension. " + 
						"Found a %d-d volume; expect size [%d,...] or [...,%d]", n, ( n - 1 ), ( n - 1 ) ) );
	}
	
    /**
     * Returns a deformation field as a {@link RandomAccessibleInterval}
     * with the vector stored in the first dimension.
     *
     * @param source
     * @return the possibly permuted deformation field
     * @throws Exception
     */
	public static final < T extends RealType< T > > RandomAccessibleInterval< T > vectorAxisFirst( RandomAccessibleInterval< T > source ) throws Exception
	{
		final int n = source.numDimensions();
		
		if ( source.dimension( 0 ) == (n - 1) )
			return source;
		else if ( source.dimension( n - 1 ) == (n - 1) )
		{
			final int[] component = new int[ n ];
			component[ n - 1 ] = 0;
			for ( int i = 0; i < n-1; ++i )
				component[ i ] = i + 1;

			return permute( source, component );
		}

		throw new Exception( 
				String.format( "Displacement fields must store vector components in the first or last dimension. " + 
						"Found a %d-d volume; expect size [%d,...] or [...,%d]", n, ( n - 1 ), ( n - 1 ) ) );
	}

    /**
     * Permutes the dimensions of a {@link RandomAccessibleInterval}
     * using the given permutation vector, where the ith value in p
     * gives destination of the ith input dimension in the output. 
     *
     * @param source the source data
     * @param p the permutation
     * @return the permuted source
     */
	public static final < T > IntervalView< T > permute( RandomAccessibleInterval< T > source, int[] p )
	{
		final int n = source.numDimensions();

		final long[] min = new long[ n ];
		final long[] max = new long[ n ];
		for ( int i = 0; i < n; ++i )
		{
			min[ p[ i ] ] = source.min( i );
			max[ p[ i ] ] = source.max( i );
		}

		final MixedTransform t = new MixedTransform( n, n );
		t.setComponentMapping( p );

		return Views.interval( new MixedTransformView< T >( source, t ), min, max );
	}

    /**
     * Returns a two element double array in which the first and second elements
     * store the minimum and maximum values of the input {@link
     * IterableInterval}, respectively.
     *
     * @param img the iterable interval
     * @return the min and max values stored in a double array
     */
	public static <T extends RealType<T>> double[] getMinMax( IterableInterval<T> img )
	{
		double min = Double.MAX_VALUE;
		double max = Double.MIN_VALUE;
		Cursor<T> c = img.cursor();
		while( c.hasNext() )
		{
			double v = Math.abs( c.next().getRealDouble());
			if( v > max )
				max = v;
			
			if( v < min )
				min = v;
		}
		return new double[]{ min, max };
	}

}
